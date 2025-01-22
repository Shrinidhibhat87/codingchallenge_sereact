"""
Function that contains the dataloader class.
To create dataloaders, inherit abstract Dataset class
Need to override __len__() and __getitem__() methods
"""

import os
from typing import Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.visualize_image import visualize_masks_on_image
from utils.visualize_point_cloud import visualize_bounding_box, visualize_point_cloud


def check_aspect(crop_range, aspect_min):
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
    return (
        (xy_aspect >= aspect_min)
        or (xz_aspect >= aspect_min)
        or (yz_aspect >= aspect_min)
    )

class RandomCuboid(object):
    """
    RandomCuboid augmentation from DepthContrast [https://arxiv.org/abs/2101.02691]
    This class is copied from the 3DDETR repository.
    This augmentation randomly crops a cuboid from the input and
    ensures that the cropped cuboid contains at least one bounding box
    """

    def __init__(
        self,
        min_points,
        aspect=0.8,
        min_crop=0.5,
        max_crop=1.0,
        box_filter_policy="center",
    ):
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

    def __call__(self, point_cloud, target_boxes, per_point_labels=None):
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
        range_xyz = np.max(point_cloud[:, 0:3], axis=0) - np.min(
            point_cloud[:, 0:3], axis=0
        )

        for _ in range(100):
            # Randomly generate crop range
            crop_range = self.min_crop + np.random.rand(3) * (
                self.max_crop - self.min_crop
            )
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
            upper_idx = (
                np.sum((point_cloud[:, 0:3] <= max_xyz).astype(np.int32), 1) == 3
            )
            lower_idx = (
                np.sum((point_cloud[:, 0:3] >= min_xyz).astype(np.int32), 1) == 3
            )

            new_pointidx = (upper_idx) & (lower_idx)

            # Ensure the cropped cuboid contains at least the minimum number of points
            if np.sum(new_pointidx) < self.min_points:
                continue

            new_point_cloud = point_cloud[new_pointidx, :]

            # Filtering policy for bounding boxes
            if self.box_filter_policy == "center":
                # Remove boxes whose center does not lie within the new_point_cloud
                new_boxes = target_boxes
                if (
                    target_boxes.sum() > 0
                ):  # Ground truth contains no bounding boxes. Common in SUNRGBD.
                    box_centers = target_boxes[:, 0:3]
                    new_pc_min_max = np.min(new_point_cloud[:, 0:3], axis=0), np.max(
                        new_point_cloud[:, 0:3], axis=0
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
        transform=None,
        debug=False,
        augment=False
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
        self.debug = debug
        self.augment = augment
        self.random_cuboid = RandomCuboid(
            min_points=30000,
            aspect=0.75,
            min_crop=0.5,
            max_crop=1.0
        )

    def __len__(self) -> int:
        """Returns the total number of data points available."""
        return len(self.folderpath)

    def __getitem__(self, index: int) -> Dict:
        """Abstract __getitem__() method that needs to be overriden.

        Args:
            index (int): Index to get the item

        Returns:
            Dict: Dictionary holding the appropriate information for that index.
        """
        subfolder = self.folderpath[index]
        subfolder_path = os.path.join(self.source_path, subfolder)

        # Load data from subdirectory. It has 4 items
        # 1) Color image named "rgb.jpg"
        # 2) 3D bounding box representation named "bbox3d.npy"
        # 3) Mask information named "mask.npy"
        # 4) Point cloud information "pc.npy"

        # 1) Load RGB image
        image = Image.open(os.path.join(subfolder_path, 'rgb.jpg')).convert('RGB')

        # 2) 3D bounding box
        bbox_3d = np.load(os.path.join(subfolder_path, 'bbox3d.npy'))

        # 3) Mask information
        mask = np.load(os.path.join(subfolder_path, 'mask.npy'))

        # 4) Point cloud
        pcd = np.load(os.path.join(subfolder_path, 'pc.npy'))
        
        # Check if asked to apply augmentation.
        if self.augment:
            pcd, bbox_3d = self.apply_augmentation(pcd, bbox_3d)

        # Once loaded, store them in a dict and return them
        data_dict = {'rgb': image, 'bbox3d': bbox_3d, 'mask': mask, 'pcd': pcd}

        return data_dict
    
    def apply_augmentation(
        self,
        pcd: np.ndarray,
        bbox_3d: np.ndarray
    ):
        """Applies augmentations to the point cloud and bounding boxes.

        Args:
            pcd (ndarray): Input point cloud data (N, 3).
            bbox_3d (ndarray): Input bounding box data (K, 8, 3).

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


    def visualize_data(self, index: int) -> None:
        sample = self[index]
        image = sample['rgb']
        pcd = sample['pcd']
        bbox3d = sample['bbox3d']
        mask = sample['mask']

        if self.debug:
            visualize_masks_on_image(image=image, masks=mask)
            visualize_point_cloud(pc_input=pcd, color_image=image)
            visualize_bounding_box(pc_input=pcd, bbox_points=bbox3d, color_image=image)
