"""
Function that contains the dataloader class.
To create dataloaders, inherit abstract Dataset class
Need to override __len__() and __getitem__() methods.
"""

import os
from typing import Dict

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.visualize_image import visualize_masks_on_image
from utils.visualize_point_cloud import (
    visualize_point_cloud,
    visualize_point_cloud_with_bounding_box,
)


def check_aspect(crop_range: np.ndarray, aspect_min: float) -> bool:
    """
    Check if the aspect ratio of the crop range is within the specified minimum aspect ratio.

    Args:
        crop_range (np.ndarray): The range of the crop in x, y, and z dimensions.
        aspect_min (float): The minimum aspect ratio to be maintained.

    Returns:
        bool: True if the aspect ratio is within the specified minimum, False otherwise.
    """
    # Calculate the aspect ratios for xy, xz, and yz planes
    xy_aspect = np.min(crop_range[:2]) / np.max(crop_range[:2])
    xz_aspect = np.min(crop_range[[0, 2]]) / np.max(crop_range[[0, 2]])
    yz_aspect = np.min(crop_range[1:]) / np.max(crop_range[1:])

    # Check if any of the aspect ratios are greater than or equal to the minimum aspect ratio
    return (xy_aspect >= aspect_min) or (xz_aspect >= aspect_min) or (yz_aspect >= aspect_min)


class RandomCuboid:
    """
    RandomCuboid augmentation from DepthContrast [https://arxiv.org/abs/2101.02691]
    This class is copied from the 3DDETR repository.
    This augmentation randomly crops a cuboid from the input and
    ensures that the cropped cuboid contains at least one bounding box
    """

    def __init__(
        self,
        min_points: int,
        aspect: float = 0.8,
        min_crop: float = 0.5,
        max_crop: float = 1.0,
        box_filter_policy: str = 'center',
    ) -> None:
        """
        Initialize the RandomCuboid object.

        Args:
            min_points (int): Minimum number of points required in the cropped cuboid.
            aspect (float, optional): Minimum aspect ratio to be maintained. Defaults to 0.8.
            min_crop (float, optional): Minimum crop size as a fraction of the original size. Defaults to 0.5.
            max_crop (float, optional): Maximum crop size as a fraction of the original size. Defaults to 1.0.
            box_filter_policy (str, optional): Policy to filter bounding boxes. Defaults to "center".
        """
        self.aspect = aspect
        self.min_crop = min_crop
        self.max_crop = max_crop
        self.min_points = min_points
        self.box_filter_policy = box_filter_policy

    def __call__(
        self, point_cloud: np.ndarray, target_boxes: np.ndarray, per_point_labels: list = None
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Apply the RandomCuboid augmentation to the input point cloud and bounding boxes.

        Args:
            point_cloud (np.ndarray): Input point cloud data.
            target_boxes (np.ndarray): Bounding boxes associated with the point cloud.
            per_point_labels (list, optional): Labels for each point in the point cloud. Defaults to None.

        Returns:
            tuple: Cropped point cloud, filtered bounding boxes, and per-point labels.
        """
        # Calculate the range of the point cloud in x, y, and z dimensions
        range_xyz = np.max(point_cloud[:, 0:3], axis=0) - np.min(point_cloud[:, 0:3], axis=0)

        for _ in range(100):
            # Randomly generate crop range
            crop_range = self.min_crop + np.random.rand(3) * (self.max_crop - self.min_crop)
            if not check_aspect(crop_range, self.aspect):
                continue

            # Randomly select a center point for the crop
            sample_center = point_cloud[np.random.choice(len(point_cloud)), 0:3]

            # Calculate the new range for the crop
            new_range = range_xyz * crop_range / 2.0

            # Determine the max and min coordinates for the crop
            max_xyz = sample_center + new_range
            min_xyz = sample_center - new_range

            # Find points within the crop range
            upper_idx = np.sum((point_cloud[:, 0:3] <= max_xyz).astype(np.int32), 1) == 3
            lower_idx = np.sum((point_cloud[:, 0:3] >= min_xyz).astype(np.int32), 1) == 3

            new_pointidx = (upper_idx) & (lower_idx)

            # Ensure the cropped cuboid contains at least the minimum number of points
            if np.sum(new_pointidx) < self.min_points:
                continue

            new_point_cloud = point_cloud[new_pointidx, :]

            # Filtering policy for bounding boxes
            if self.box_filter_policy == 'center':
                # Remove boxes whose center does not lie within the new_point_cloud
                new_boxes = target_boxes
                if (
                    target_boxes.sum() > 0
                ):  # Ground truth contains no bounding boxes. Common in SUNRGBD.
                    box_centers = target_boxes[:, 0:3]
                    new_pc_min_max = (
                        np.min(new_point_cloud[:, 0:3], axis=0),
                        np.max(new_point_cloud[:, 0:3], axis=0),
                    )
                    keep_boxes = np.logical_and(
                        np.all(box_centers >= new_pc_min_max[0], axis=1),
                        np.all(box_centers <= new_pc_min_max[1], axis=1),
                    )
                    if keep_boxes.sum() == 0:
                        # Current data augmentation removes all boxes in the pointcloud. Fail!
                        continue
                    new_boxes = target_boxes[keep_boxes]
                if per_point_labels is not None:
                    new_per_point_labels = [x[new_pointidx] for x in per_point_labels]
                else:
                    new_per_point_labels = None
                # If we are here, all conditions are met. Return boxes
                return new_point_cloud, new_boxes, new_per_point_labels

        # Fallback
        return point_cloud, target_boxes, per_point_labels


class SereactDataloader(Dataset):
    """Sereact specific dataloader class with data augmentation for 3D bounding box localization.

    Args:
        Dataset (_type_): Torch datasetclass. Override abstract methods
    """

    def __init__(
        self,
        source_path: str,
        transform: callable = None,
        debug: bool = False,
        augment: bool = False,
    ) -> None:
        """Constructor method

        Args:
            source_path (str): Path to where the data is stored
            transform (_type_, optional): Transforms to applied to sample. Defaults to None.
            debug (bool, optional): If True, enables debug visualization. Defaults to False.
            augment (bool, optional): If True, applies data augmentation. Defaults to False.
        """
        self.source_path = source_path
        self.tranform = transform
        self.folderpath = os.listdir(source_path)
        self.augment = augment
        self.debug = debug
        self.random_cuboid = RandomCuboid(min_points=30000, aspect=0.75, min_crop=0.5, max_crop=1.0)

    def __len__(self) -> int:
        """Returns the total number of data points available."""
        return len(self.folderpath)

    def normalize_pointcloud(self, points: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, float]:
        """Normalize point cloud to zero mean and unit sphere.

        Args:
            points (np.ndarray): Point cloud of shape (3, H, W)

        Returns:
            tuple: (normalized_points, centroid, scale_factor)
        """
        # Reshape to (N, 3)
        points = points.reshape(3, -1).T

        # Center the point cloud
        centroid = np.mean(points, axis=0)
        points = points - centroid

        # Scale to unit sphere
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / max_dist

        # (N, 3)
        return points, centroid, max_dist

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Abstract __getitem__() method that needs to be overriden.

        Args:
            index (int): Index to get the item

        Returns:
            Dict[str, torch.Tensor]: Dictionary holding the appropriate information for that index.
        """
        subfolder = self.folderpath[index]
        subfolder_path = os.path.join(self.source_path, subfolder)

        # Load data from subdirectory. It has 4 items
        # 1) Color image named "rgb.jpg"
        # 2) 3D bounding box representation named "bbox3d.npy"
        # 3) Mask information named "mask.npy"
        # 4) Point cloud information "pc.npy"

        # 1) Load RGB image
        image = np.array(Image.open(os.path.join(subfolder_path, 'rgb.jpg')).convert('RGB'))
        # Convert this to a torch tensor for further usage
        rgb_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        # Normalize the image
        if rgb_tensor.max() > 1:
            rgb_tensor = rgb_tensor / 255.0

        # 2) 3D bounding box
        bbox_3d = np.load(os.path.join(subfolder_path, 'bbox3d.npy'))
        # Have a tensor form of the bounding box
        bbox_3d_tensor = torch.tensor(bbox_3d, dtype=torch.float32)

        # 3) Mask information
        mask = np.load(os.path.join(subfolder_path, 'mask.npy'))

        # 4) Point cloud
        pcd = np.load(os.path.join(subfolder_path, 'pc.npy'))
        # Have a tensor form of the point cloud
        pcd_tensor = torch.from_numpy(pcd.reshape(3, -1).T).float()

        # Check if asked to apply augmentation.
        if self.augment:
            pcd, bbox_3d = self.apply_augmentation(pcd, bbox_3d)

        # Normalize point cloud
        pcd_normalized, centroid, max_dist = self.normalize_pointcloud(pcd)

        # Convert to tensor (already in correct shape)
        pcd_tensor = torch.from_numpy(pcd_normalized).float()

        # Normalize bbox coordinates (shape: (K, 8, 3))
        bbox_3d_normalized = bbox_3d.copy()
        for i in range(bbox_3d.shape[0]):
            for j in range(bbox_3d.shape[1]):
                bbox_3d_normalized[i, j] = (bbox_3d[i, j] - centroid) / max_dist

        # Convert to tensor
        bbox_3d_tensor = torch.from_numpy(bbox_3d_normalized).float()

        # Store in data dict
        if self.debug:
            data_dict = {
                'rgb': image,
                'bbox3d': bbox_3d,
                'mask': mask,
                'pcd': pcd,
                'pcd_tensor': pcd_tensor,
                'bbox3d_tensor': bbox_3d_tensor,
                'normalization_params': {
                    'centroid': torch.from_numpy(centroid).float(),
                    'scale': torch.tensor(max_dist).float(),
                },
            }
        else:
            data_dict = {
                'rgb_tensor': rgb_tensor,
                'pcd_tensor': pcd_tensor,
                'bbox3d_tensor': bbox_3d_tensor,
                'point_cloud_dims_min': pcd_normalized.reshape(-1, 3).min(axis=0)[:3],
                'point_cloud_dims_max': pcd_normalized.reshape(-1, 3).max(axis=0)[:3],
                'normalization_params': {
                    'centroid': torch.from_numpy(centroid).float(),
                    'scale': torch.tensor(max_dist).float(),
                },
            }

        return data_dict

    def apply_augmentation(
        self, pcd: np.ndarray, bbox_3d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Applies augmentations to the point cloud and bounding boxes.

        Args:
            pcd (np.ndarray): Input point cloud data (N, 3).
            bbox_3d (np.ndarray): Input bounding box data (K, 8, 3).

        Returns:
            tuple: Augmented point cloud and bounding boxes.
        """

        # First augmentation: Flip along X-axis
        if np.random.rand() > 0.5:
            pcd[:, 0] = -pcd[:, 0]
            bbox_3d[:, :, 0] = -bbox_3d[:, :, 0]

        # Second augmentation: Rotate around Z-axis
        if np.random.rand() > 0.5:
            # Rotate -30° to +30°
            angle = (np.random.rand() * np.pi / 3) - np.pi / 6
            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            # Rotate the points based on the rotation matrix
            pcd[:, :3] = np.dot(pcd[:, :3], rotation_matrix.T)

            # Reshape the bbox and apply the same rotation/augmentation
            bbox_3d = np.dot(bbox_3d.reshape(-1, 3), rotation_matrix.T).reshape(-1, 8, 3)

        # Third augmentation: Scale the point cloud
        scale_factor = np.random.uniform(0.8, 1.2)
        pcd[:, :3] *= scale_factor
        bbox_3d[:, :, :3] *= scale_factor

        # Fourth augmentation: Jitter that adds small random noise to the point cloud
        jitter = np.random.normal(0, 0.01, pcd[:, :3].shape)
        pcd[:, :3] += jitter

        # Fifth augmentation: Use the RandomCuboid augmentation from DepthContrast
        if np.random.rand() > 0.5:
            pcd, bbox_3d, _ = self.random_cuboid(pcd, bbox_3d)

        return pcd, bbox_3d

    def get_datasets(self, test_size: float = 0.2, random_seed: int = 40) -> tuple:
        """Splits the dataset into training and testing datasets.

        Args:
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 40.

        Returns:
            tuple: Training and testing datasets.
        """
        # Get the indices from the folderpath
        indices = list(range(len(self.folderpath)))
        # Split using the sklearn library
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_seed
        )
        # Return the training and testing datasets
        train_dataset = torch.utils.data.Subset(self, train_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)

        return train_dataset, test_dataset

    def visualize_data(self, index: int) -> None:
        sample = self[index]
        image = Image.fromarray(sample['rgb'])
        pcd = sample['pcd']
        bbox3d = sample['bbox3d']
        mask = sample['mask']

        visualize_masks_on_image(image=image, masks=mask)
        visualize_point_cloud(pc_input=pcd, color_image=image)
        visualize_point_cloud_with_bounding_box(pc_input=pcd, bbox_points=bbox3d, color_image=image)
