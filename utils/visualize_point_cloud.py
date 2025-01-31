"""
This file consists of 2 utility files.
1. visualize_point_cloud: Function used to visualize the pointcloud representation.
2. visualize_bounding_box: Function used to visualize the pointcloud with bbox in the pc.
"""
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import open3d as o3d
from PIL import Image


def visualize_point_cloud(
    pc_input: Union[str, npt.NDArray], color_image: Optional[Union[str, npt.NDArray]] = None
) -> None:
    """Visualize Point Cloud.

    Args:
        pc_input (Union[str, npt.NDArray]): path to pointcloud file (.npy) or np array
        color_image (Optional[npt.NDArray], optional): path to color image (.npy) or np array. Defaults to None.

    Raises:
        TypeError: If invalid input type is passed.
    """
    if isinstance(pc_input, np.ndarray):
        pc = pc_input
    elif isinstance(pc_input, str):
        pc = np.load(pc_input)
    else:
        raise TypeError(
            f'Invalid pointcloud input. Expected str or numpy type. Got {type(pc_input)}'
        )
    # If the color image is passed, check for the type of the color image passed
    if color_image is not None:
        if isinstance(color_image, Image.Image):
            pil_image = color_image
        elif isinstance(color_image, str):
            pil_image = Image.open(color_image).convert('RGB')
        else:
            raise TypeError(
                f'Invalid image input. Expected str or PIL image type. Got {type(pil_image)}'
            )
    # Convert the incoming pointcloud np_array into structured form
    if len(pc.shape) == 3:
        pcd_n_3 = pc.transpose(1, 2, 0).reshape(-1, 3)
    if color_image is not None:
        image_array = np.asarray(pil_image)
        normalized_array = image_array.astype(np.float64) / 255.0
        np_array_n_3 = np.asarray(normalized_array).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_n_3)
    pcd.colors = o3d.utility.Vector3dVector(np_array_n_3)
    o3d.visualization.draw_geometries([pcd])


def visualize_bounding_box(
    pc_input: Union[str, npt.NDArray],
    bbox_points: Union[str, npt.NDArray],
    color_image: Optional[Union[str, npt.NDArray]] = None,
) -> None:
    """Visualize pointcloud with bounding box

    Args:
        pc_input (Union[str, npt.NDArray]): path to pointcloud file (.npy) or np array
        bbox_points (Union[str, npt.NDArray]): path to bbox file (.npy) or np array
        color_image (Optional[npt.NDArray], optional): path to color image (.npy) or np array. Defaults to None.

    Raises:
        TypeError: If invalid pointcloud points are passed.
        TypeError: If invalid bbox points are passed.
    """
    if isinstance(pc_input, np.ndarray):
        pc = pc_input
    elif isinstance(pc_input, str):
        pc = np.load(pc_input)
    else:
        raise TypeError(
            f'Invalid pointcloud input. Expected str or numpy type. Got {type(pc_input)}'
        )

    if isinstance(bbox_points, np.ndarray):
        bbox = bbox_points
    elif isinstance(bbox_points, str):
        bbox = np.load(bbox_points)
    else:
        raise TypeError(
            f'Invalid boundingbox input. Expected str or numpy type. Got {type(pc_input)}'
        )

    # If the color image is passed, check for the type of the color image passed
    if color_image is not None:
        if isinstance(color_image, Image.Image):
            pil_image = color_image
        elif isinstance(color_image, str):
            pil_image = Image.open(color_image).convert('RGB')
        else:
            raise TypeError(
                f'Invalid image input. Expected str or PIL image type. Got {type(pil_image)}'
            )
    if len(pc.shape) == 3:
        pc = pc.transpose(1, 2, 0).reshape(-1, 3)
    pc = pc.astype(np.float64)
    if color_image is not None:
        image_array = np.asarray(pil_image)
        normalized_array = image_array.astype(np.float64) / 255.0
        np_array_n_3 = np.asarray(normalized_array).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if color_image is not None:
        pcd.colors = o3d.utility.Vector3dVector(np_array_n_3)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    bbox = bbox.astype(np.float64)
    for bounding_box in bbox:
        bbox_3d_points = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(bounding_box)
        )
        vis.add_geometry(bbox_3d_points)

    vis.run()
    vis.destroy_window()
