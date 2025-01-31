"""
Python file that contains the training method, Logger details and also the optimizer
"""

import datetime
import os
import time

import torch
from omegaconf import DictConfig

import wandb
from utils.mean_iou_evaluation import IoUEvaluator


class Trainer:
    """
    A class to handle the training and validation of a model, including checkpointing and logging.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        validate_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): The device (CPU or GPU) used for computation.
        checkpoint_dir (str): Directory to save checkpoints.
        start_epoch (int): The starting epoch for training.
        max_epochs (int): The maximum number of epochs to train.
        best_iou (float): The best IoU metric achieved during validation.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        validate_dataloader: torch.utils.data.DataLoader,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
    ) -> None:
        """
        Initializes the Trainer class with the necessary components for training.

        Args:
            cfg (DictConfig): Configuration object containing training parameters.
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            criterion (torch.nn.Module): The loss function used for training.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            validate_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            device (torch.device): The device (CPU or GPU) used for computation.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg
        self.checkpoint_dir = cfg.checkpoint_dir
        self.start_epoch = cfg.start_epoch
        self.max_epochs = cfg.max_epochs
        self.best_iou = -1.0

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, f'bbox_detection_{timestamp}')
            os.makedirs(self.checkpoint_dir)

    def save_checkpoint(self, epoch: int, val_metrics: dict, filename: str = None) -> None:
        """
        Saves a checkpoint of the model, optimizer, and other training states.

        Args:
            epoch (int): The current epoch number.
            val_metrics (dict): Validation metrics to be saved.
            filename (str, optional): The name of the checkpoint file. If None, a default name is used.
        """
        if filename is None:
            filename = f'checkpoint_{epoch:04d}.pth'
        checkpoint_name = os.path.join(self.checkpoint_dir, filename)

        sd = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'args': self.cfg,
            'best_val_metrics': val_metrics,
        }
        torch.save(sd, checkpoint_name)

    def train_one_epoch(
        self, epoch: int, iou_evaluator: IoUEvaluator, log_interval: int = 10
    ) -> dict:
        """
        Trains the model for one epoch.

        Args:
            epoch (int): The current epoch number.
            iou_evaluator (IoUEvaluator): Instance of the IoUEvaluator class for evaluating bounding box predictions.
            log_interval (int, optional): Interval for logging metrics to the console and wandb.

        Returns:
            dict: Training metrics including mean loss and IoU metrics.
        """
        self.model.train()
        epoch_loss = 0.0
        total_batches = len(self.train_dataloader)
        start_time = time.time()

        # Reset IoU evaluator for the epoch
        iou_evaluator.reset()
        batch_time_accum = 0.0

        for batch_idx, batch in enumerate(self.train_dataloader, start=1):
            batch_start_time = time.time()

            # Move input data to the specified device
            inputs = [obj.to(self.device) for obj in batch['pcd_tensor']]
            gt_bboxes = [obj.to(self.device) for obj in batch['bbox3d_tensor']]
            pcd_dims_min = [obj.to(self.device) for obj in batch['point_cloud_dims_min']]
            pcd_dims_max = [obj.to(self.device) for obj in batch['point_cloud_dims_max']]

            # Enable anomaly detection for debugging
            torch.autograd.set_detect_anomaly(True)

            # Zero gradients at the start of the batch
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                inputs,
                point_cloud_dims_min=pcd_dims_min,
                point_cloud_dims_max=pcd_dims_max,
            )

            # Unpack single output from the list
            output = outputs[0]
            gt_bbox = gt_bboxes[0]

            # Get predictions
            pred_boxes = output['outputs']

            # Compute loss
            loss, loss_dict, assignments = self.criterion(pred_boxes, gt_bbox)

            # Backpropagation
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1, norm_type=2.0)

            # Parameter update
            self.optimizer.step()

            # Learning rate scheduler step
            self.scheduler.step()

            # Update epoch loss
            epoch_loss += loss.item()

            # Update IoU evaluator
            predicted_bboxes_matched, gt_bboxes_matched = (
                self.get_predicted_and_gt_boxes_from_assignments(
                    pred_boxes=pred_boxes, assignments=assignments, gt_bbox=gt_bbox
                )
            )

            # Update IoU evaluator
            iou_evaluator.update(predicted_bboxes_matched, gt_bboxes_matched)

            # End timer and accumulate batch time
            batch_time_accum += time.time() - batch_start_time

            # Logging at intervals
            if batch_idx % log_interval == 0 or batch_idx == total_batches:
                mean_loss = epoch_loss / batch_idx
                avg_batch_time = batch_time_accum / log_interval
                eta = avg_batch_time * (total_batches - batch_idx)
                eta_str = str(datetime.timedelta(seconds=int(eta)))

                print(
                    f'Epoch [{epoch}]: Batch [{batch_idx}/{total_batches}], '
                    f'Loss: {mean_loss:.4f}, ETA: {eta_str}'
                )

                # Log to Weights and Biases
                wandb.log(
                    {
                        'epoch': epoch,
                        'batch': batch_idx,
                        'loss': mean_loss,
                        'batch_time': avg_batch_time,
                    }
                )

                # Reset batch time accumulator
                batch_time_accum = 0.0

        # Compute IoU metrics for the epoch
        metrics = iou_evaluator.compute_metrics()
        mean_loss = epoch_loss / total_batches

        # Log final epoch metrics to Weights and Biases
        wandb.log(
            {
                'epoch': epoch,
                'mean_loss': mean_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'mean_iou': metrics['mean_iou'],
                'epoch_time': time.time() - start_time,
                **{f'iou@{t}': acc for t, acc in metrics['threshold_accuracy'].items()},
            }
        )

        # Print final metrics
        print(f'Epoch [{epoch}] Complete: Mean Loss: {mean_loss:.4f}, IoU Metrics: {metrics}')

        return {'mean_loss': mean_loss, 'iou_metrics': metrics}

    @torch.no_grad()
    def validate(self, iou_evaluator: IoUEvaluator) -> dict:
        """
        Validates the model using the validation dataset.

        Args:
            iou_evaluator (IoUEvaluator): IoU evaluator to compute intersection-over-union metrics.

        Returns:
            dict: Validation metrics including IoU and optional loss metrics.
        """
        self.model.eval()
        iou_evaluator.reset()
        metrics = {'iou': {}, 'loss': 0.0}
        total_loss = 0.0
        num_batches = len(self.validate_dataloader)
        time_per_batch = []

        print('Starting validation...')

        for batch_idx, batch_data in enumerate(self.validate_dataloader):
            start_time = time.time()

            # Move input data to the specified device
            inputs = [obj.to(self.device) for obj in batch_data['pcd_tensor']]
            gt_bboxes = [obj.to(self.device) for obj in batch_data['bbox3d_tensor']]
            pcd_dims_min = [obj.to(self.device) for obj in batch_data['point_cloud_dims_min']]
            pcd_dims_max = [obj.to(self.device) for obj in batch_data['point_cloud_dims_max']]

            # Forward pass
            # inputs = {"point_clouds": batch_data["pcd"]}
            outputs = self.model(
                inputs,
                point_cloud_dims_min=pcd_dims_min,
                point_cloud_dims_max=pcd_dims_max,
            )

            # Unpack the output from the list
            output = outputs[0]
            gt_bbox = gt_bboxes[0]

            # Get predictions
            pred_boxes = output['outputs']

            # Compute loss (if criterion is provided)
            if self.criterion:
                loss, loss_dict, assignments = self.criterion(outputs=pred_boxes, targets=gt_bbox)

                total_loss += loss.item()

            # Update IoU metrics
            if iou_evaluator:
                predicted_bboxes_matched, gt_bboxes_matched = (
                    self.get_predicted_and_gt_boxes_from_assignments(
                        pred_boxes=pred_boxes, assignments=assignments, gt_bbox=gt_bbox
                    )
                )
                """
                # Just for visualization purpose right now.
                visualize_bounding_box(
                    pc_input=inputs[0].cpu().detach().numpy(),
                    bbox_points=gt_bboxes_matched,
                    color_image=None,
                )

                visualize_bounding_box(
                    pc_input=inputs[0].cpu().detach().numpy(),
                    bbox_points=predicted_bboxes_matched,
                    color_image=None,
                )
                """

                # Update IoU evaluator
                iou_evaluator.update(predicted_bboxes_matched, gt_bboxes_matched)

            # Track time per batch
            time_per_batch.append(time.time() - start_time)

            # Print progress
            print(
                f'Batch {batch_idx + 1}/{num_batches} '
                f'Time per batch: {sum(time_per_batch) / len(time_per_batch):.2f}s'
            )

        # Compute IoU metrics for the validation
        metrics = iou_evaluator.compute_metrics()
        mean_loss = total_loss / num_batches

        # Log final validation metrics to Weights and Biases
        wandb.log(
            {
                'mean_loss': mean_loss,
                'mean_iou': metrics['mean_iou'],
                **{f'iou@{t}': acc for t, acc in metrics['threshold_accuracy'].items()},
            }
        )

        # Print final metrics
        print('Validation completed!')
        print(f'IoU Metrics: {metrics}')
        print(f'Average Loss: {mean_loss:.4f}')

        return {'mean_loss': mean_loss, 'iou_metrics': metrics}

    def get_predicted_and_gt_boxes_from_assignments(
        self, pred_boxes: dict, assignments: dict, gt_bbox: torch.Tensor
    ) -> tuple:
        """
        Extracts matched predicted and ground truth bounding boxes based on assignments.

        Args:
            pred_boxes (dict): Dictionary containing predicted bounding boxes.
            assignments (dict): Dictionary containing assignment indices.
            gt_bbox (torch.Tensor): Ground truth bounding boxes.

        Returns:
            tuple: Matched predicted and ground truth bounding boxes.
        """
        # Get the predicted and gt indices
        matched_predicted_indices = assignments['assignments'][0][0]
        matched_gt_indices = assignments['assignments'][0][1]

        # Move to CPU
        matched_predicted_indices = matched_predicted_indices.cpu().detach().numpy()
        matched_gt_indices = matched_gt_indices.cpu().detach().numpy()

        # Index-slicing to get matched boxes
        predicted_bboxes_matched = pred_boxes['box_corners'][0, matched_predicted_indices]
        gt_bboxes_matched = gt_bbox[matched_gt_indices]

        # Move to CPU
        predicted_bboxes_matched = predicted_bboxes_matched.cpu().detach().numpy()
        gt_bboxes_matched = gt_bboxes_matched.cpu().detach().numpy()

        return predicted_bboxes_matched, gt_bboxes_matched

    def train(self) -> None:
        """
        Main training loop for training the model and saving checkpoints.
        """
        # Display training details
        print(f'Model: {self.model}')
        print(f'Optimizer: {self.optimizer}')
        print(f'Scheduler: {self.scheduler}')
        print(f'Criterion: {self.criterion}')
        print(f'Training starts at epoch {self.start_epoch} and ends at epoch {self.max_epochs}')
        print(f'Iterations per training epoch: {len(self.train_dataloader)}')
        print(f'Iterations per validation epoch: {len(self.validate_dataloader)}')

        # Main training loop
        for epoch in range(self.start_epoch, self.max_epochs):
            print(f'Starting epoch {epoch}/{self.max_epochs}')

            # Train for one epoch
            train_metrics = self.train_one_epoch(
                epoch=epoch,
                iou_evaluator=IoUEvaluator(iou_thresholds=[0.25, 0.5]),
            )

            # Log training metrics
            wandb.log({'epoch': epoch, **train_metrics})

            # Save latest checkpoint
            self.save_checkpoint(
                epoch=epoch,
                val_metrics=None,
                filename='checkpoint_latest.pth',
            )

            print(f'Checkpoint saved for epoch {epoch}')

            # Run validation
            print(f'Running validation for epoch {epoch}')
            val_metrics = self.validate(
                iou_evaluator=IoUEvaluator(iou_thresholds=[0.25, 0.5]),
            )

            # Log validation metrics
            wandb.log({'epoch': epoch, **val_metrics})

            # Save best checkpoint based on validation IoU
            current_iou = val_metrics['iou_metrics']['mean_iou']
            if current_iou > self.best_iou:
                self.best_iou = current_iou
                self.save_checkpoint(
                    epoch=epoch,
                    val_metrics=val_metrics,
                    filename='checkpoint_best.pth',
                )
                print(f'Best checkpoint saved at epoch {epoch}')

        # Save final evaluation results
        print('Training completed. Saving final evaluation results.')
        final_eval_path = os.path.join(self.checkpoint_dir, 'final_eval.txt')
        with open(final_eval_path, 'w') as eval_file:
            eval_file.write('Training Completed.\n')
            eval_file.write(f'Best Validation Metrics: {self.best_iou}\n')

        # Save final model weights
        final_weights_path = os.path.join(self.checkpoint_dir, 'final_weights.pth')
        torch.save(self.model.state_dict(), final_weights_path)
        print(f'Final model weights saved at {final_weights_path}')
