"""
Main file.
"""

import argparse
import os
import torch
import sys
import numpy as np
from typing import List
from torch.utils.data import DataLoader

from dataloader import SereactDataloader
from models.detr3d.model_3ddetr import build_3ddetr_model
from losses.loss_3ddetr import build_loss_object
from trainer.trainer import train
from utils.miscellaneous import worker_init_fn



def set_device():
    """Set the device to CUDA if GPU available."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_optimizer(args, model):
    """
    Build an AdamW optimizer with optional weight decay filtering for biases and parameters with shape length 1.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - filter_biases_wd (bool): Whether to filter out biases and parameters with shape length 1 from weight decay.
            - weight_decay (float): The weight decay to apply to parameters.
            - base_lr (float): The base learning rate for the optimizer.
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
        if args.filter_biases_wd and (len(param.shape) == 1 or name.endswith("bias")):
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    # Create parameter groups with appropriate weight decay settings
    if args.filter_biases_wd:
        param_groups = [
            {"params": params_without_decay, "weight_decay": 0.0},
            {"params": params_with_decay, "weight_decay": args.weight_decay},
        ]
    else:
        param_groups = [
            {"params": params_with_decay, "weight_decay": args.weight_decay},
        ]

    # Build the AdamW optimizer with the specified parameter groups and learning rate
    optimizer = torch.optim.AdamW(param_groups, lr=args.base_lr)
    
    return optimizer


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing the folder path.
    """
    parser = argparse.ArgumentParser(description='Dataloader parser.')
    
    ############ Input folder path ############
    parser.add_argument(
        'input_folder_path',
        type=str,
        help='Path to the folder containing subfolders with required files',
    )

    ############ Training Parameters ############
    parser.add_argument('--seed', type=int, default=40, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')    
    
    ############ Testing Parameters ############
    parser.add_argument('--test_only', type=bool, default=False, help='Flag to run test only')

    ############ Model parameters ############
    ########## Preencoder and Encoder parameters ##########
    parser.add_argument('--encoder_dim', default=256, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--encoder_nheads", default=4, type=int)
    parser.add_argument("--encoder_ffn_dim", default=128, type=int)
    parser.add_argument("--encoder_dropout", default=0.1, type=float)
    parser.add_argument("--encoder_activation", default='relu', type=str)
    parser.add_argument("--encoder_num_layers", default=3, type=int)
    parser.add_argument("--encoder_type", default='vanilla', type=str)
    parser.add_argument("--preencoder_npoints", default=2048, type=int)
    # Maybe need some parameters for the preencoder
    ########## Decoder parameters ##########
    parser.add_argument('--decoder_dim', default=256, type=int)
    parser.add_argument("--decoder_nhead", default=4, type=int)
    parser.add_argument("--decoder_ffn_dim", default=256, type=int)
    parser.add_argument("--decoder_dropout", default=0.1, type=float)
    parser.add_argument("--decoder_num_layers", default=3, type=int)
    ########## Other parameters ##########
    parser.add_argument("--position_embedding", default='fourier', type=str, choices=['sine', 'fourier'])
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument("--num_queries", default=256, type=int)
    parser.add_argument("--num_angular_bins", default=12, type=int)
    
    ############ Optimizer variables ############
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ############ Loss related parameters ############
    ##### Set Loss #####
    parser.add_argument("--matcher_cost_giou", default=2.0, type=float)
    parser.add_argument("--matcher_cost_center", default=0.0, type=float)
    parser.add_argument("--matcher_cost_objectness", default=0.0, type=float)
    ##### Loss Weights #####
    parser.add_argument("--loss_giou_weight", default=0.0, type=float)
    parser.add_argument("--loss_center_weight", default=5.0, type=float)
    parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
    parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
    parser.add_argument("--loss_size_weight", default=1.0, type=float)
    parser.add_argument("--loss_no_object_weight", default=0.2, type=float)
    
    ############ Miscellaneous parameters ############
    parser.add_argument('--debug', type=bool, default=False, help='Flag to degug and visualize')
    parser.add_argument('--ds_number', type=int, default=13, help='Data set index to visualize')

    ############ Checkpoint directory ############
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save/load checkpoints')

    return parser.parse_args()


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


def main() -> None:
    """
    Main function to set up and run the 3DDETR training pipeline.

    This function:
    - Parses input arguments.
    - Validates the folder structure.
    - Initializes the dataset and dataloaders.
    - Builds the model, loss object, and optimizer.
    - Handles training or testing based on the arguments.
    """
    
    args = parse_args()
    try:
        folder_path = validate_folder_structure(args.input_folder_path)
        print('Valid input folder path')
        
        # Initialize the dataset for training 
        dataset = SereactDataloader(source_path=folder_path)
        if args.debug:
            dataset.visualize_data(args.ds_number)
        
        # Set the device accordingly
        DEVICE = set_device()
        
        # Since the model training requires GPU, raise error if on CPU
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            raise RuntimeError("GPU not available. Please check if CUDA is enabled.")
        
        # Set random seeds for reproducibility
        seed = args.seed
        # Consistent for numpy
        np.random.seed(seed)
        # Consistent for pytorch
        torch.manual_seed(seed)
        # Consistent for GPU
        torch.cuda.manual_seed_all(seed)
        
        # Setup and build the 3DDETR model
        print(f"Setting and building the 3DDETR model")
        model = build_3ddetr_model(args)
        model.to(DEVICE)
        
        # Setup and build the loss object
        print(f"Setting up the loss object")
        criterion = build_loss_object(args) # Need to add config here or change things

        # Create dataset and dataloaders
        print(f"Initializing the dataset and dataloaders")
        train_dataset, test_dataset = dataset.get_datasets() # Need to add this function

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # Setup optimizer
        print(f"Setting up the optimizer")
        optimizer = build_optimizer(args, model)

        # Resume from checkpoint if applicable.
        start_epoch = 0
        if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
            print(f"Loading checkpoint from {args.checkpoint_dir}...")
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, "checkpoint.pth"))
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {start_epoch}.")

        # Train or test the model.
        if args.test_only:
            print(f"Running in test-only mode")
            train(
                model=model,
                criterion=None,  # Loss is not required for testing.
                optimizer=None,
                train_loader=None,
                test_loader=test_loader,
                start_epoch=start_epoch,
                num_epochs=0,  # No training epochs for testing.
                checkpoint_dir=None,
            )
        else:
            print(f"Starting training...")
            train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                test_loader=test_loader,
                start_epoch=start_epoch,
                num_epochs=args.epochs,
                checkpoint_dir=args.checkpoint_dir,
            )

    except (NotADirectoryError, FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
