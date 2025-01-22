"""
Python file that contains the training method, Logger details and also the optimizer
"""

import numpy as np
import torch
import os
import time
import datetime
import wandb

from utils.mean_iou_evaluation import IoUEvaluator


def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    epoch,
    args,
    best_val_metrics,
    filename=None,
):
    # Might have to add a line for it to be distributed
    if filename is None:
        filename = f"checkpoint_{epoch:04d}.pth"
    checkpoint_name = os.path.join(checkpoint_dir, filename)

    sd = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "best_val_metrics": best_val_metrics,
    }
    torch.save(sd, checkpoint_name)

def train_one_epoch(
    model,
    optimizer,
    metric,
    data_loader,
    device,
    epoch,
    iou_evaluator,
    log_interval=10,
):
    """
    Train the model for one epoch.

    Args:
        model: The model being trained.
        optimizer: Optimizer for training.
        metric: Loss function for training.
        data_loader: DataLoader providing training data batches.
        device: The device (CPU or GPU) used for computation.
        epoch: Current training epoch.
        iou_evaluator: Instance of the IoUEvaluator class for evaluating bounding box predictions.
        log_interval: Interval for logging metrics to the console and wandb.

    Returns:
        dict: Training metrics including mean loss and IoU metrics.
    """
    model.train()
    epoch_loss = 0.0
    total_batches = len(data_loader)
    start_time = time.time()

    # Reset IoU evaluator for the epoch
    iou_evaluator.reset()

    for batch_idx, batch in enumerate(data_loader, start=1):
        # Move input data to the specified device
        inputs = batch["input"].to(device)
        gt_boxes = batch["gt_boxes"].to(device)

        # Forward pass
        optimizer.zero_grad()
        # Possibly would have to add the input as a form of dictionary
        outputs = model(inputs)
        pred_boxes = outputs["box_corners"]

        # Compute loss
        loss = metric(pred_boxes, gt_boxes)
        loss.backward()
        optimizer.step()

        # Update epoch loss
        epoch_loss += loss.item()

        # Update IoU evaluator
        iou_evaluator.update(pred_boxes.cpu().detach().numpy(), gt_boxes.cpu().detach().numpy())

        # Logging at intervals
        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            mean_loss = epoch_loss / batch_idx
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / batch_idx) * (total_batches - batch_idx)
            eta_str = str(datetime.timedelta(seconds=int(eta)))

            print(
                f"Epoch [{epoch}]: Batch [{batch_idx}/{total_batches}], "
                f"Loss: {mean_loss:.4f}, ETA: {eta_str}"
            )

            # Log to Weights and Biases
            wandb.log({
                "epoch": epoch,
                "batch": batch_idx,
                "loss": mean_loss,
                "batch_time": elapsed_time / batch_idx,
            })

    # Compute IoU metrics for the epoch
    metrics = iou_evaluator.compute_metrics()
    mean_loss = epoch_loss / total_batches

    # Log final epoch metrics to Weights and Biases
    wandb.log({
        "epoch": epoch,
        "mean_loss": mean_loss,
        "mean_iou": metrics["mean_iou"],
        **{f"iou@{t}": acc for t, acc in metrics["threshold_accuracy"].items()},
    })

    # Print final metrics
    print(f"Epoch [{epoch}] Complete: Mean Loss: {mean_loss:.4f}, IoU Metrics: {metrics}")

    return {"mean_loss": mean_loss, "iou_metrics": metrics}


@torch.no_grad()
def validate(
    model,
    data_loader,
    device,
    criterion=None,
    iou_evaluator=None,
):
    """
    Validation function to evaluate the model using IoU metrics.

    Args:
        model: Model to be evaluated.
        data_loader: DataLoader for the validation dataset.
        device: Torch device (CPU or GPU) to run the model on.
        criterion: Optional loss function for validation.
        iou_evaluator: IoU evaluator to compute intersection-over-union metrics.

    Returns:
        metrics: Dictionary containing IoU and optional loss metrics.
    """
    # Switch the model to evaluation mode
    model.eval()

    # Initialize metrics and timers
    metrics = {"iou": {}, "loss": 0.0}
    total_loss = 0.0
    num_batches = len(data_loader)
    time_per_batch = []

    print("Starting validation...")

    for batch_idx, batch_data in enumerate(data_loader):
        start_time = time.time()

        # Move data to the target device
        for key in batch_data:
            batch_data[key] = batch_data[key].to(device)

        # Forward pass
        inputs = {"point_clouds": batch_data["point_clouds"]}
        outputs = model(inputs)

        # Compute loss (if criterion is provided)
        if criterion:
            loss, _ = criterion(outputs, batch_data)
            total_loss += loss.item()

        # Update IoU metrics
        if iou_evaluator:
            pred_boxes = outputs["outputs"]["boxes"] # Assuming predictions include 'boxes'
            gt_boxes = batch_data["boxes"] # Assuming ground truths include 'boxes'
            # iou_evaluator.update(outputs["outputs"], batch_data)
            iou_evaluator.update(pred_boxes.cpu().detach().numpy(), gt_boxes.cpu().detach().numpy())

        # Track time per batch
        time_per_batch.append(time.time() - start_time)

        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(
                f"Batch {batch_idx + 1}/{num_batches} "
                f"Time per batch: {sum(time_per_batch) / len(time_per_batch):.2f}s"
            )

    # Compute final IoU metrics
    if iou_evaluator:
        metrics["iou"] = iou_evaluator.compute_metrics()

    # Average loss over all batches
    if criterion:
        metrics["loss"] = total_loss / num_batches

    print("Validation completed.")
    print(f"IoU Metrics: {metrics['iou']}")
    if criterion:
        print(f"Average Loss: {metrics['loss']:.4f}")

    return metrics


def train(
    args,
    model,
    optimizer,
    criterion,
    config,
    data_loader,
    device,
):
    """
    Main training loop for training a model and saving checkpoints.

    Args:
        args: Command-line arguments with training configurations.
        model: Model to be trained.
        optimizer: Optimizer used for updating model weights.
        criterion: Loss function used for training.
        config: Configuration file or dictionary with model/data parameters.
        data_loader: Dictionary with "train" and "val" DataLoaders.
        device: Torch device (CPU or GPU) to run the model on.
    """
    # Get iterations per epoch for train and validation
    num_iterations_per_epoch = len(data_loader["train"])
    num_iterations_per_validation = len(data_loader["val"])

    # Display training details
    print(f"Model: {model}")
    print(f"Optimizer: {optimizer}")
    print(f"Training starts at epoch {args.start_epoch} and ends at epoch {args.max_epoch}")
    print(f"Iterations per training epoch: {num_iterations_per_epoch}")
    print(f"Iterations per validation epoch: {num_iterations_per_validation}")

    # Paths for saving final evaluation results and weights
    final_eval_path = os.path.join(args.checkpoint_dir, "final_eval.txt")
    final_weights_path = os.path.join(args.checkpoint_dir, "final_weights.pth")

    # Check if training was completed previously
    if os.path.isfile(final_eval_path):
        print(f"Training already completed. Found file: {final_eval_path}. Skipping...")
        return

    # Initialize Weights and Biases
    wandb.init(project="model_training", config=args)
    wandb.watch(model, log="all")

    # Main training loop
    for epoch in range(args.start_epoch, args.max_epoch):
        print(f"Starting epoch {epoch}/{args.max_epoch}")

        # Train for one epoch
        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=data_loader["train"],
            device=device,
            epoch=epoch,
            iou_evaluator=IoUEvaluator(iou_thresholds=[0.25, 0.5]),
        )

        # Log training metrics
        wandb.log({"epoch": epoch, **train_metrics})

        # Save latest checkpoint
        save_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_metrics=None,
            filename="checkpoint_latest.pth",
        )

        print(f"Checkpoint saved for epoch {epoch}")

        # Validation at specified intervals
        if epoch % args.eval_every_epoch == 0 or epoch == args.max_epoch - 1:
            print(f"Running validation for epoch {epoch}")
            val_metrics = validate(
                model=model,
                data_loader=data_loader["val"],
                device=device,
                criterion=criterion,
                iou_evaluator=IoUEvaluator(iou_thresholds=[0.25, 0.5]),
            )

            # Log validation metrics
            wandb.log({"epoch": epoch, **val_metrics})

            # Save best checkpoint based on validation IoU
            if (
                not args.best_val_metrics
                or val_metrics["mean_iou"] > args.best_val_metrics["mean_iou"]
            ):
                args.best_val_metrics = val_metrics
                save_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_metrics=args.best_val_metrics,
                    filename="checkpoint_best.pth",
                )
                print(f"Best checkpoint saved at epoch {epoch}")

    # Save final evaluation results
    print("Training completed. Saving final evaluation results.")
    with open(final_eval_path, "w") as eval_file:
        eval_file.write("Training Completed.\n")
        eval_file.write(f"Best Validation Metrics: {args.best_val_metrics}\n")

    # Save final model weights
    torch.save(model.state_dict(), final_weights_path)
    print(f"Final model weights saved at {final_weights_path}")
