"""
Python file that contains some miscellaneous functions that for augmentation and model training.
"""

from typing import List

import numpy as np
import torch


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize the random seed for a worker process.

    This function is used to ensure that each worker process has a different random seed,
    which is important for data loading in parallel.

    Args:
        worker_id (int): The ID of the worker process.
    """
    # Set the random seed for numpy based on the current random state and worker ID
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def shift_scale_points(
    pred_xyz: torch.Tensor, src_range: list[torch.Tensor], dst_range: list[torch.Tensor] = None
) -> torch.Tensor:
    """
    Shift and scale points from source range to destination range.

    Args:
        pred_xyz (torch.Tensor): Predicted coordinates of shape (B, N, 3).
        src_range (list[torch.Tensor]): Source range as a list of two tensors, each of shape (B, 3), representing min and max XYZ coordinates.
        dst_range (list[torch.Tensor], optional): Destination range as a list of two tensors, each of shape (B, 3), representing min and max XYZ coordinates. Default is None.

    Returns:
        torch.Tensor: Transformed coordinates of shape (B, N, 3).
    """
    # The source range is shaped (3,). So need to reshape this.
    batch_size = pred_xyz.shape[0]
    # Reshaped to (1, 3)
    src_min = src_range[0].float().unsqueeze(0)
    src_max = src_range[1].float().unsqueeze(0)
    src_range = [src_min, src_max]

    if dst_range is None:
        # Default destination range is [0, 1] for each coordinate
        dst_min = torch.zeros((batch_size, 3), device=src_range[0].device, dtype=src_range[0].dtype)
        dst_max = torch.ones((batch_size, 3), device=src_range[0].device, dtype=src_range[0].dtype)
        dst_range = [dst_min, dst_max]

    if pred_xyz.ndim == 4:
        # Adjust ranges for batched input
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    # Ensure input dimensions match
    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    # Calculate differences between min and max coordinates for source and destination ranges
    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]

    # Shift and scale points from source range to destination range
    prop_xyz = (((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff) + dst_range[0][
        :, None, :
    ]

    return prop_xyz


def scale_points(pred_xyz: torch.Tensor, mult_factor: torch.Tensor) -> torch.Tensor:
    """
    Scales the given points by a multiplication factor using PyTorch operations.

    Parameters:
    pred_xyz (torch.Tensor): A Tensor of shape (..., 3) representing the points to be scaled.
    mult_factor (torch.Tensor): A Tensor of shape (3,) representing the scaling factors.

    Returns:
    torch.Tensor: A Tensor of the same shape as pred_xyz with the points scaled by the multiplication factor.
    """
    # Get the number of dimensions in pred_xyz
    num_dims = pred_xyz.dim()
    # Reshape mult_factor to have the same number of dimensions as pred_xyz, with the last dimension being 3
    for _ in range(num_dims - 1):
        mult_factor = mult_factor.unsqueeze(0)
    # Multiply pred_xyz by mult_factor
    scaled_xyz = pred_xyz * mult_factor
    return scaled_xyz


# Link: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/bba1f6156371fbabf02bf4c47062dfde21a32b46/log/classification/pointnet2_ssg_wo_normals/pointnet2_utils.py#L63
# Discussion: https://github.com/rusty1s/pytorch_cluster/issues/102#issuecomment-834017428
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # Initialize centroids
    distance = torch.ones(B, N).to(device) * 1e10  # Initialize distances to a large value
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(
        device
    )  # Randomly select the first point
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # Batch indices for indexing

    for i in range(npoint):
        centroids[:, i] = farthest  # Assign the farthest point as a centroid
        centroid = xyz[batch_indices, farthest, :].view(
            B, 1, 3
        )  # Get the coordinates of the farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)  # Compute squared distances from the centroid
        mask = dist < distance  # Find points closer than the current farthest distance
        distance[mask] = dist[mask]  # Update distances
        farthest = torch.max(distance, -1)[1]  # Select the next farthest point

    return centroids


def collate_fn(batch: List[dict]) -> dict:
    """
    Collate function to be used with DataLoader to stack the data in the batch.

    Args:
        batch (List): A list of dictionaries containing the data and label.

    Returns:
        dict: A dictionary containing the stacked data and label.
    """
    # Load PCD tensors
    pcd_tensors = [item['pcd_tensor'] for item in batch]
    # Load Bbox tensors
    bbox_tensors = [item['bbox3d_tensor'] for item in batch]
    # Point cloud min and max dimensions
    pcd_min = [torch.from_numpy(item['point_cloud_dims_min']) for item in batch]
    pcd_max = [torch.from_numpy(item['point_cloud_dims_max']) for item in batch]
    # RGB tensor image
    rgb_tensors = [item['rgb_tensor'] for item in batch]

    return {
        # Can't stack here because of different number of points
        'pcd_tensor': pcd_tensors,
        'bbox3d_tensor': bbox_tensors,
        'point_cloud_dims_min': pcd_min,
        'point_cloud_dims_max': pcd_max,
        'rgb_tensor': rgb_tensors,
    }


def move_to_device(data: dict | torch.Tensor, device: torch.device) -> dict | torch.Tensor:
    """
    Move the data to the specified device.

    Args:
        data (dict | torch.Tensor): A dictionary containing the data to be moved to the device.
        device (torch.device): The device to move the data to.

    Returns:
        dict | torch.Tensor: A dictionary containing the data moved to the device.
    """
    if isinstance(data, dict):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data
