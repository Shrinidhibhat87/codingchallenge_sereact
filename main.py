"""
Main file.
"""

import os
import sys
from typing import List

import hydra
import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import wandb
from dataloader import SereactDataloader
from losses.loss_3ddetr import LossFunction
from models.detr3d.model_3ddetr import build_3ddetr_model
from trainer.trainer import Trainer
from utils.low_precision_conversion import convert_model_to_low_precision
from utils.mean_iou_evaluation import IoUEvaluator
from utils.miscellaneous import collate_fn, worker_init_fn


def set_device() -> torch.device:
    """Set the device to CUDA if GPU available."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_optimizer(cfg_opt: DictConfig, model: torch.nn.Module) -> torch.optim.AdamW:
    """
    Build an AdamW optimizer with optional weight decay filtering for biases and parameters with shape length 1.

    Args:
        cfg_opt (DictConfig): A configuration object containing optimizer parameters.
        model (torch.nn.Module): The model containing parameters to optimize.

    Returns:
        torch.optim.AdamW: An AdamW optimizer configured with the specified parameters and weight decay settings.
    """
    # Initialize lists to hold parameters with and without weight decay
    params_with_decay = []
    params_without_decay = []

    # Iterate over model parameters
    for name, param in model.named_parameters():
        # Skip parameters that do not require gradients
        if param.requires_grad is False:
            continue
        # Filter out biases and parameters with shape length 1 if specified
        if cfg_opt.filter_biases_wd and (len(param.shape) == 1 or name.endswith('bias')):
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    # Create parameter groups with appropriate weight decay settings
    if cfg_opt.filter_biases_wd:
        param_groups = [
            {'params': params_without_decay, 'weight_decay': 0.0},
            {'params': params_with_decay, 'weight_decay': cfg_opt.weight_decay},
        ]
    else:
        param_groups = [
            {'params': params_with_decay, 'weight_decay': cfg_opt.weight_decay},
        ]

    # Build the AdamW optimizer with the specified parameter groups and learning rate
    optimizer = torch.optim.AdamW(param_groups, lr=cfg_opt.base_lr)

    return optimizer


def validate_folder_structure(folder_path: str) -> str:
    """
    Validate the folder structure to ensure it contains subdirectories, each with specific required files.

    Args:
        folder_path (str): Path to the main folder to be validated.

    Raises:
        NotADirectoryError: If the provided path is not a directory.
        ValueError: If the folder does not contain any subdirectories.
        FileNotFoundError: If any required file is missing in the subdirectories.

    Returns:
        str: The validated folder path.
    """
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f'The path {folder_path} does not exist or is not a directory.')

    # List all subdirectories in the given folder
    subdirectories = [
        os.path.join(folder_path, subdir)
        for subdir in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, subdir))
    ]

    # Ensure there is at least one subdirectory
    if not subdirectories:
        raise ValueError(f'The folder {folder_path} does not contain any subdirectories.')

    # Files that must be present in each subdirectory
    required_files: List[str] = ['rgb.jpg', 'bbox3d.npy', 'mask.npy', 'pc.npy']

    for subdir in subdirectories:
        for required_file in required_files:
            file_path = os.path.join(subdir, required_file)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(
                    f'The required file {required_file} is missing in the subdirectory {subdir}.'
                )

    return folder_path


@hydra.main(version_base=None, config_path='config', config_name='base_training')
def main(cfg: DictConfig) -> None:
    """
    Main function to set up and run the 3DDETR training pipeline.

    This function:
    - Parses input arguments.
    - Validates the folder structure.
    - Initializes the dataset and dataloaders.
    - Builds the model, loss object, and optimizer.
    - Handles training or testing based on the arguments.
    """

    try:
        folder_path = validate_folder_structure(cfg.input_folder_path)
        print('Valid input folder path')

        # Initialize the dataset for training
        dataset = SereactDataloader(source_path=folder_path, debug=cfg.debug.enable)
        if cfg.debug.enable:
            dataset.visualize_data(cfg.debug.ds_number)

        # Set the device accordingly
        DEVICE = set_device()

        # Since the model training requires GPU, raise error if on CPU
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
            print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        else:
            raise RuntimeError('GPU not available. Please check if CUDA is enabled.')

        # Set random seeds for reproducibility
        seed = cfg.seed
        # Consistent for numpy
        np.random.seed(seed)
        # Consistent for pytorch
        torch.manual_seed(seed)
        # Consistent for GPU
        torch.cuda.manual_seed_all(seed)

        # Setup and build the 3DDETR model
        print('Setting and building the 3DDETR model')
        model = build_3ddetr_model(cfg.model)
        print(f'The model looks like: {model}')
        model.to(DEVICE)

        # Setup and build the loss object
        print('Setting up the loss object')
        loss_module = LossFunction(cfg_loss=cfg.loss)
        # criterion = build_loss_object(cfg.loss)

        # Create dataset and dataloaders
        print('Initializing the dataset and dataloaders')
        train_dataset, test_dataset = dataset.get_datasets(test_size=cfg.test_split)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            worker_init_fn=worker_init_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
        )

        # Setup optimizer
        print('Setting up the optimizer')
        optimizer = build_optimizer(cfg.optimizer, model)

        # Create a learning rate scheduler
        if cfg.optimizer.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
        elif cfg.optimizer.lr_scheduler == 'cosine_warmup':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.optimizer.warm_lr_epochs, T_mult=2
            )
        else:
            raise ValueError(f'Invalid learning rate scheduler: {cfg.optimizer.lr_scheduler}')

        # Build a Trainer object
        trainer = Trainer(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            criterion=loss_module,
            train_dataloader=train_loader,
            validate_dataloader=test_loader,
            scheduler=scheduler,
            device=DEVICE,
        )

        # Initialize weights and biases logger
        # Hydra uses omegaconf to parse the config, so we need to convert it to a dictionary
        config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project='3D Bounding box prediction', config=config_dict)
        wandb.watch(model, log='all')

        # Resume from checkpoint if applicable.
        start_epoch = 0
        if cfg.checkpoint_dir and os.path.exists(cfg.checkpoint_dir + 'checkpoint.pth'):
            print(f'Loading checkpoint from {cfg.checkpoint_dir}...')
            # Need to also check if the model can be loaded with pre-trained weights
            checkpoint = torch.load(os.path.join(cfg.checkpoint_dir, 'checkpoint.pth'))

            # Load the model with weights from checkpoint
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)

            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Resume from the last epoch
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f'Resumed from epoch {start_epoch}.')
            wandb.config.update({'checkpoint_loaded': True, 'pretrained_weights': False})
        else:
            print('No checkpoint directory with checkpoint path is provided.')
            wandb.config.update({'checkpoint_loaded': False})

        # Check if there is a valid pre-trained weights path and if so, load the weights
        if (
            start_epoch == 0
            and cfg.model.pretrained_weights_path
            and os.path.isfile(cfg.model.pretrained_weights_path)
        ):
            print(f'Loaded pretrained weights from {cfg.model.pretrained_weights_path}')
            model.load_state_dict(
                torch.load(cfg.model.pretrained_weights_path, map_location=DEVICE)
            )
            wandb.config.update({'pretrained_weights': True})
            wandb.config.update({'pretrained_weights_path': cfg.model.pretrained_weights_path})
        else:
            if start_epoch == 0:
                print('No pretrained weights provided. Training from scratch.')
            wandb.config.update({'pretrained_weights': False})

        # Train or test/validate the model.
        if cfg.valid_only:
            print('Validation only...')
            trainer.validate(iou_evaluator=IoUEvaluator())
        else:
            print('Starting training...')
            trainer.train()

        if cfg.export_model:
            print('Start of conversion to low precision formats')
            convert_model_to_low_precision(cfg, model, DEVICE)
            print('Model conversion successul')

    except (NotADirectoryError, FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
