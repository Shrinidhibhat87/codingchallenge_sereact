"""
This file consists of 2 utility files.
1. visualize_point_cloud: Function used to visualize the pointcloud representation.
2. visualize_point_cloud_with_bounding_box: Function used to visualize the pointcloud with bbox in the pc.
3. visualize_bounding_boxes: Function used to visualize the predicted and ground truth bounding boxes in 3D space.
4. visualize_gui_pointcloud_with_bounding_boxes: Function used to visualize the pointcloud with bounding boxes in a GUI.
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import open3d as o3d
import open3d.visualization.gui as gui
import torch
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
        pc = pc.transpose(1, 2, 0).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if color_image is not None and pil_image is not None:
        image_array = np.asarray(pil_image)
        normalized_array = image_array.astype(np.float64) / 255.0
        np_array_n_3 = np.asarray(normalized_array).reshape(-1, 3)
        pcd.colors = o3d.utility.Vector3dVector(np_array_n_3)

    o3d.visualization.draw_geometries([pcd])


def visualize_point_cloud_with_bounding_box(
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

    # Only process color image if it exists
    np_array_n_3 = None
    if color_image is not None and pil_image is not None:
        image_array = np.asarray(pil_image)
        normalized_array = image_array.astype(np.float64) / 255.0
        np_array_n_3 = np.asarray(normalized_array).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if np_array_n_3 is not None:
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


def visualize_bounding_boxes(
    pred_boxes: npt.NDArray,
    gt_boxes: npt.NDArray,
    pred_color: tuple = (1, 0, 0),
    gt_color: tuple = (0, 1, 0),
) -> None:
    """Visualize predicted and ground truth bounding boxes in 3D space.

    Args:
        pred_boxes (npt.NDArray): Predicted bounding boxes with shape (num_boxes, 8, 3)
        gt_boxes (npt.NDArray): Ground truth bounding boxes with shape (num_boxes, 8, 3)
        pred_color (tuple, optional): RGB color for predicted boxes. Defaults to (1, 0, 0).
        gt_color (tuple, optional): RGB color for ground truth boxes. Defaults to (0, 1, 0).

    Raises:
        ValueError: If input arrays don't have the expected shape (num_boxes, 8, 3)
    """
    # Validate input shapes
    for boxes, name in [(pred_boxes, 'pred_boxes'), (gt_boxes, 'gt_boxes')]:
        if len(boxes.shape) != 3 or boxes.shape[1:] != (8, 3):
            raise ValueError(f'{name} must have shape (num_boxes, 8, 3), got {boxes.shape}')

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Create and add predicted boxes (red)
    for box_corners in pred_boxes:
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(box_corners)
        )
        bbox.color = pred_color
        vis.add_geometry(bbox)

    # Create and add ground truth boxes (green)
    for box_corners in gt_boxes:
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(box_corners)
        )
        bbox.color = gt_color
        vis.add_geometry(bbox)

    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # Set default camera view
    view_control = vis.get_view_control()
    # Looking from front
    view_control.set_front([0, 0, -1])
    # Up direction
    view_control.set_up([0, -1, 0])
    view_control.set_zoom(0.7)

    vis.run()
    vis.destroy_window()


def visualize_gui_pointcloud_with_bounding_boxes(
    pcd_points: npt.NDArray,
    rgb_tensor: torch.Tensor,
    predicted_bboxes_matched: npt.NDArray,
    gt_bboxes_matched: npt.NDArray,
) -> None:
    # Transfor the tensor into the correct shape
    rgb_points = (rgb_tensor.permute(1, 2, 0)).detach().cpu().numpy()
    rgb_points = rgb_points.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_points)

    # Initialize the application
    app = gui.Application.instance
    app.initialize()

    # Create the visualizer
    vis = o3d.visualization.O3DVisualizer('Open3D - Bounding Box Visualization', 1024, 768)
    vis.show_settings = True

    # Add the point cloud
    vis.add_geometry('PointCloud', pcd)

    # For predicted bounding boxes
    for i, bounding_box in enumerate(predicted_bboxes_matched):
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(bounding_box)
        )
        bbox.color = (1, 0, 0)
        vis.add_geometry(f'predicted_bbox_{i}', bbox)

    # For ground truth bounding boxes
    for i, bounding_box in enumerate(gt_bboxes_matched):
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(bounding_box)
        )
        bbox.color = (0, 1, 0)
        vis.add_geometry(f'gt_bbox_{i}', bbox)

    # Add a legend
    legend_pos = np.array([-5, 5, 0])

    # Add red sphere for predicted bounding box
    red_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    red_sphere.translate(legend_pos)
    red_sphere.paint_uniform_color([1, 0, 0])
    vis.add_geometry('RedSphere', red_sphere)

    # Add text for predicted bounding box
    vis.add_3d_label(legend_pos + np.array([0.5, 0, 0]), 'Predicted')

    # Add green sphere for ground truth bounding box
    green_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    green_sphere.translate(legend_pos + np.array([0, -1, 0]))
    green_sphere.paint_uniform_color([0, 1, 0])
    vis.add_geometry('GreenSphere', green_sphere)

    # Add text for ground truth bounding box
    vis.add_3d_label(legend_pos + np.array([0.5, -1, 0]), 'Ground Truth')

    # Run the visualizer
    app.add_window(vis)
    app.run()
