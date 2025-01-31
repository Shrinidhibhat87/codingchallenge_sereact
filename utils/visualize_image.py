"""
This file consists of 2 utility files.
1. visualize_image: Function used to visualize the image.
2. visualize_masks_on_image: Function used to visualize the bbox in the image.
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image


def visualize_image(image: Union[str, Image.Image]) -> None:
    """
    Visualize image.

    Args:
        image (Union[str, Image.Image]): path to image or the PIL image format.

    Raises:
        TypeError: For invalid input types.
    """
    if isinstance(image, Image.Image):
        show_image = image
    elif isinstance(image, str):
        image_ = Image.open(image).convert('RGB')
        show_image = image_
    else:
        raise TypeError(f'Invalid image input. Expected str or PIL image type. Got {type(image)}')

    show_image.show()


def visualize_masks_on_image(
    image: Union[str, Image.Image], masks: Union[str, npt.NDArray]
) -> None:
    """
    Visualize image and the binary mask of the image.

    Args:
        image (Union[str, Image.Image]): path to image or the PIL image format.
        masks (Union[str, npt.NDArray]): path to mask or numpy image format.

    Raises:
        TypeError: For invalid input image types.
        TypeError: For invalid masks type.
    """
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    elif isinstance(image, str):
        image_ = Image.open(image).convert('RGB')
        image_array = np.array(image_)
    else:
        raise TypeError(f'Invalid image input. Expected str or PIL image type. Got {type(image)}')

    if isinstance(masks, np.ndarray):
        maskarray = masks
    elif isinstance(masks, str):
        maskarray = np.load(masks)
    else:
        raise TypeError(f'Invalid boundingbox input. Expected str or numpy type. Got {type(masks)}')

    # Create a unique colormap for each mask
    num_masks = len(masks)
    colors = plt.cm.get_cmap('tab20', num_masks)
    unique_colors = [colors(i) for i in range(num_masks)]

    # Create a figure with two subplots.
    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image on the left.
    axes[0].imshow(image_array)
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    # Plot the masks on the image on the left of the subplots.
    axes[1].imshow(image_array)
    for i, mask in enumerate(maskarray):
        # Overlay the mask on the image
        plt.contourf(mask, levels=[0.5, 1.5], colors=[unique_colors[i]], alpha=0.5, cmap=None)
    axes[1].axis('off')
    axes[1].set_title('Image with masks')

    plt.show()
