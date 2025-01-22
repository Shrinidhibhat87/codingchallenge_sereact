"""
Python file that contains some miscellaneous functions that for augmentation and model training.
"""
import torch


def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    Shift and scale points from source range to destination range.

    Args:
        pred_xyz (Tensor): Predicted coordinates of shape (B, N, 3).
        src_range (list of Tensor): Source range as a list of two tensors, each of shape (B, 3), representing min and max XYZ coordinates.
        dst_range (list of Tensor, optional): Destination range as a list of two tensors, each of shape (B, 3), representing min and max XYZ coordinates. Default is None.

    Returns:
        Tensor: Transformed coordinates of shape (B, N, 3).
    """
    if dst_range is None:
        # Default destination range is [0, 1] for each coordinate
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

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
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]

    return prop_xyz


def scale_points(pred_xyz, mult_factor):
    """
    Scales the given points by a multiplication factor.

    Parameters:
    pred_xyz (numpy.ndarray): A numpy array of shape (..., 3) representing the points to be scaled.
    mult_factor (numpy.ndarray): A numpy array of shape (...) representing the scaling factors.

    Returns:
    numpy.ndarray: A numpy array of the same shape as pred_xyz with the points scaled by the multiplication factor.
    """
    if pred_xyz.ndim == 4:
        mult_factor = mult_factor[:, None]
    scaled_xyz = pred_xyz * mult_factor[:, None, :]
    return scaled_xyz


# Link: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/bba1f6156371fbabf02bf4c47062dfde21a32b46/log/classification/pointnet2_ssg_wo_normals/pointnet2_utils.py#L63
# Discussion: https://github.com/rusty1s/pytorch_cluster/issues/102#issuecomment-834017428
def farthest_point_sample(xyz, npoint):
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
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # Randomly select the first point
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # Batch indices for indexing

    for i in range(npoint):
        centroids[:, i] = farthest  # Assign the farthest point as a centroid
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # Get the coordinates of the farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)  # Compute squared distances from the centroid
        mask = dist < distance  # Find points closer than the current farthest distance
        distance[mask] = dist[mask]  # Update distances
        farthest = torch.max(distance, -1)[1]  # Select the next farthest point

    return centroids
