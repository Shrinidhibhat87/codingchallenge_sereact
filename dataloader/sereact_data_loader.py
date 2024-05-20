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

from utils.visualize_image import visualize_image, visualize_masks_on_image
from utils.visualize_point_cloud import visualize_bounding_box, visualize_point_cloud


class SereactDataloader(Dataset):
    """Sereact specific dataloader class

    Args:
        Dataset (_type_): Torch datasetclass. Override abstract methods
    """

    def __init__(self, source_path: str, transform=None, debug=False) -> None:
        """Constructor method

        Args:
            source_path (str): Path to where the data is stored
            transform (_type_, optional): Transforms to applied to sample.
                Defaults to None.
        """
        self.source_path = source_path
        self.tranform = transform
        self.folderpath = os.listdir(source_path)
        self.debug = debug

    def __len__(self) -> int:
        """Abstract __len__() method that needs to be overriden

        Returns:
            int: The total number of data points available
        """
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
        image_path = os.path.join(subfolder_path, 'rgb.jpg')
        image = Image.open(image_path).convert('RGB')

        # 2) 3D bounding box
        bbox_path = os.path.join(subfolder_path, 'bbox3d.npy')
        bbox_3d = np.load(bbox_path)

        # 3) Mask information
        mask_path = os.path.join(subfolder_path, 'mask.npy')
        mask = np.load(mask_path)

        # 4) Point cloud
        pcd_path = os.path.join(subfolder_path, 'pc.npy')
        pcd = np.load(pcd_path)

        # Once loaded, store them in a dict and return them
        data_dict = {'rgb': image, 'bbox3d': bbox_3d, 'mask': mask, 'pcd': pcd}

        return data_dict

    def visualize_data(self, index: int) -> None:
        sample = self[index]
        image = sample['rgb']
        pcd = sample['pcd']
        bbox3d = sample['bbox3d']
        mask = sample['mask']

        if self.debug:
            visualize_image(image=image)
            visualize_masks_on_image(image=image, masks=mask)
            visualize_point_cloud(pc_input=pcd, color_image=image)
            visualize_bounding_box(pc_input=pcd, bbox_points=bbox3d, color_image=image)
