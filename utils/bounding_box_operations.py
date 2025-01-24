"""
Utility file for Bouning box related operations.
"""

import torch
import numpy as np

from typing import List

@torch.jit.ignore
def to_list_1d(arr) -> List[float]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr

@torch.jit.ignore
def to_list_3d(arr) -> List[List[List[float]]]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr

def computeIntersection(cp1, cp2, s, e):
    dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
    dp = [s[0] - e[0], s[1] - e[1]]
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    
    return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

def inside(cp1, cp2, p):
    return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])


def polygon_clip_unnest(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
        subjectPolygon: a list of (x,y) 2d points, any polygon.
        clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
        **points have to be counter-clockwise ordered**

    Return:
        a list of (x,y) vertex point for the intersection polygon.
    """
    outputList = [subjectPolygon[x] for x in range(subjectPolygon.shape[0])]
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList.copy()
        outputList.clear()
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(cp1, cp2, e):
                if not inside(cp1, cp2, s):
                    outputList.append(computeIntersection(cp1, cp2, s, e))
                outputList.append(e)
            elif inside(cp1, cp2, s):
                outputList.append(computeIntersection(cp1, cp2, s, e))
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            break
    return outputList

def box_intersection(rect1, rect2, non_rot_inter_areas, nums_k2, inter_areas, approximate):
    """
    rect1 - B x K1 x 8 x 3 matrix of box corners
    rect2 - B x K2 x 8 x 3 matrix of box corners
    non_rot_inter_areas - intersection areas of boxes 
    """
    
    B = rect1.shape[0]
    K1 = rect1.shape[1]
    K2 = rect2.shape[1]

    for b in range(B):
        for k1 in range(K1):
            for k2 in range(K2):
                if k2 >= nums_k2[b]:
                    break
                
                if approximate and non_rot_inter_areas[b][k1][k2] == 0:
                    continue
                
                # Compute volume of intersection
                inter = polygon_clip_unnest(rect1[b, k1], rect2[b, k2])
                ninter = len(inter)
                if ninter > 0:  # there is some intersection between the boxes
                    xs = np.array([x[0] for x in inter]).astype(dtype=float)
                    ys = np.array([x[1] for x in inter]).astype(dtype=float)
                    inter_areas[b, k1, k2] = 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))

def enclosing_box3d_vol(corners1, corners2):
    """
    volume of enclosing axis-aligned box
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape) == 4
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners2.shape[2] == 8
    assert corners2.shape[3] == 3
    EPS = 1e-6

    corners1 = corners1.clone()
    corners2 = corners2.clone()
    # flip Y axis, since it is negative
    corners1[:, :, :, 1] *= -1
    corners2[:, :, :, 1] *= -1

    al_xmin = torch.min(
        torch.min(corners1[:, :, :, 0], dim=2).values[:, :, None],
        torch.min(corners2[:, :, :, 0], dim=2).values[:, None, :],
    )
    al_ymin = torch.max(
        torch.max(corners1[:, :, :, 1], dim=2).values[:, :, None],
        torch.max(corners2[:, :, :, 1], dim=2).values[:, None, :],
    )
    al_zmin = torch.min(
        torch.min(corners1[:, :, :, 2], dim=2).values[:, :, None],
        torch.min(corners2[:, :, :, 2], dim=2).values[:, None, :],
    )
    al_xmax = torch.max(
        torch.max(corners1[:, :, :, 0], dim=2).values[:, :, None],
        torch.max(corners2[:, :, :, 0], dim=2).values[:, None, :],
    )
    al_ymax = torch.min(
        torch.min(corners1[:, :, :, 1], dim=2).values[:, :, None],
        torch.min(corners2[:, :, :, 1], dim=2).values[:, None, :],
    )
    al_zmax = torch.max(
        torch.max(corners1[:, :, :, 2], dim=2).values[:, :, None],
        torch.max(corners2[:, :, :, 2], dim=2).values[:, None, :],
    )

    diff_x = torch.abs(al_xmax - al_xmin)
    diff_y = torch.abs(al_ymax - al_ymin)
    diff_z = torch.abs(al_zmax - al_zmin)
    vol = diff_x * diff_y * diff_z
    return vol


def box3d_vol_tensor(corners):
    EPS = 1e-6
    reshape = False
    B, K = corners.shape[0], corners.shape[1]
    if len(corners.shape) == 4:
        # batch x prop x 8 x 3
        reshape = True
        corners = corners.view(-1, 8, 3)
    a = torch.sqrt(
        (corners[:, 0, :] - corners[:, 1, :]).pow(2).sum(dim=1).clamp(min=EPS)
    )
    b = torch.sqrt(
        (corners[:, 1, :] - corners[:, 2, :]).pow(2).sum(dim=1).clamp(min=EPS)
    )
    c = torch.sqrt(
        (corners[:, 0, :] - corners[:, 4, :]).pow(2).sum(dim=1).clamp(min=EPS)
    )
    vols = a * b * c
    if reshape:
        vols = vols.view(B, K)
    return vols


def generalized_box3d_iou_cython(
    corners1: torch.Tensor,
    corners2: torch.Tensor,
    nums_k2: torch.Tensor,
    rotated_boxes: bool = True,
    return_inter_vols_only: bool = False,
):
    """
    Input:
        corners1: torch Tensor (B, K1, 8, 3), assume up direction is negative Y
        corners2: torch Tensor (B, K2, 8, 3), assume up direction is negative Y
        Assumes that the box is only rotated along Z direction
    Returns:
        B x K1 x K2 matrix of generalized IOU by approximating the boxes to be axis aligned
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape) == 4
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == corners2.shape[2]
    assert corners1.shape[3] == corners2.shape[3]

    B, K1 = corners1.shape[0], corners1.shape[1]
    _, K2 = corners2.shape[0], corners2.shape[1]

    # # box height. Y is negative, so max is torch.min
    ymax = torch.min(corners1[:, :, 0, 1][:, :, None], corners2[:, :, 0, 1][:, None, :])
    ymin = torch.max(corners1[:, :, 4, 1][:, :, None], corners2[:, :, 4, 1][:, None, :])
    height = (ymax - ymin).clamp(min=0)
    EPS = 1e-8

    idx = torch.arange(start=3, end=-1, step=-1, device=corners1.device)
    idx2 = torch.tensor([0, 2], dtype=torch.int64, device=corners1.device)
    rect1 = corners1[:, :, idx, :]
    rect2 = corners2[:, :, idx, :]
    rect1 = rect1[:, :, :, idx2]
    rect2 = rect2[:, :, :, idx2]

    lt = torch.max(rect1[:, :, 1][:, :, None, :], rect2[:, :, 1][:, None, :, :])
    rb = torch.min(rect1[:, :, 3][:, :, None, :], rect2[:, :, 3][:, None, :, :])
    wh = (rb - lt).clamp(min=0)
    non_rot_inter_areas = wh[:, :, :, 0] * wh[:, :, :, 1]
    non_rot_inter_areas = non_rot_inter_areas.view(B, K1, K2)
    if nums_k2 is not None:
        for b in range(B):
            non_rot_inter_areas[b, :, nums_k2[b] :] = 0

    enclosing_vols = enclosing_box3d_vol(corners1, corners2)

    # vols of boxes
    vols1 = box3d_vol_tensor(corners1).clamp(min=EPS)
    vols2 = box3d_vol_tensor(corners2).clamp(min=EPS)

    sum_vols = vols1[:, :, None] + vols2[:, None, :]

    # filter malformed boxes
    good_boxes = (enclosing_vols > 2 * EPS) * (sum_vols > 4 * EPS)

    if rotated_boxes:
        inter_areas = np.zeros((B, K1, K2), dtype=np.float32)
        rect1 = rect1.cpu().numpy().astype(np.float32)
        rect2 = rect2.cpu().numpy().astype(np.float32)
        nums_k2_np = nums_k2.cpu().detach().numpy().astype(np.int32)
        non_rot_inter_areas_np = (
            non_rot_inter_areas.cpu().detach().numpy().astype(np.float32)
        )
        box_intersection(
            rect1, rect2, non_rot_inter_areas_np, nums_k2_np, inter_areas, True
        )
        inter_areas = torch.from_numpy(inter_areas)
    else:
        inter_areas = non_rot_inter_areas

    inter_areas = inter_areas.to(corners1.device)
    ### gIOU = iou - (1 - sum_vols/enclose_vol)
    inter_vols = inter_areas * height
    if return_inter_vols_only:
        return inter_vols

    union_vols = (sum_vols - inter_vols).clamp(min=EPS)
    ious = inter_vols / union_vols
    giou_second_term = -(1 - union_vols / enclosing_vols)
    gious = ious + giou_second_term
    gious *= good_boxes
    if nums_k2 is not None:
        mask = torch.zeros((B, K1, K2), device=height.device, dtype=torch.float32)
        for b in range(B):
            mask[b, :, : nums_k2[b]] = 1
        gious *= mask
    return gious

"""
# Results in a runtime error, which is why commenting this code out for now.
def generalized_box3d_iou_tensor(
    corners1: torch.Tensor,
    corners2: torch.Tensor,
    nums_k2: torch.Tensor,
    rotated_boxes: bool = True,
    return_inter_vols_only: bool = False,
):
    # Need comment lines here
    Input:
        corners1: torch Tensor (B, K1, 8, 3), assume up direction is negative Y
        corners2: torch Tensor (B, K2, 8, 3), assume up direction is negative Y
        Assumes that the box is only rotated along Z direction
    Returns:
        B x K1 x K2 matrix of generalized IOU by approximating the boxes to be axis aligned
    # Need comment lines here
    assert len(corners1.shape) == 4
    assert len(corners2.shape) == 4
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == corners2.shape[2]
    assert corners1.shape[3] == corners2.shape[3]

    B, K1 = corners1.shape[0], corners1.shape[1]
    _, K2 = corners2.shape[0], corners2.shape[1]

    # # box height. Y is negative, so max is torch.min
    ymax = torch.min(corners1[:, :, 0, 1][:, :, None], corners2[:, :, 0, 1][:, None, :])
    ymin = torch.max(corners1[:, :, 4, 1][:, :, None], corners2[:, :, 4, 1][:, None, :])
    height = (ymax - ymin).clamp(min=0)
    EPS = 1e-8

    idx = torch.arange(start=3, end=-1, step=-1, device=corners1.device)
    idx2 = torch.tensor([0, 2], dtype=torch.int64, device=corners1.device)
    rect1 = corners1[:, :, idx, :]
    rect2 = corners2[:, :, idx, :]
    rect1 = rect1[:, :, :, idx2]
    rect2 = rect2[:, :, :, idx2]

    lt = torch.max(rect1[:, :, 1][:, :, None, :], rect2[:, :, 1][:, None, :, :])
    rb = torch.min(rect1[:, :, 3][:, :, None, :], rect2[:, :, 3][:, None, :, :])
    wh = (rb - lt).clamp(min=0)
    non_rot_inter_areas = wh[:, :, :, 0] * wh[:, :, :, 1]
    non_rot_inter_areas = non_rot_inter_areas.view(B, K1, K2)
    if nums_k2 is not None:
        for b in range(B):
            non_rot_inter_areas[b, :, nums_k2[b] :] = 0

    enclosing_vols = enclosing_box3d_vol(corners1, corners2)

    # vols of boxes
    vols1 = box3d_vol_tensor(corners1).clamp(min=EPS)
    vols2 = box3d_vol_tensor(corners2).clamp(min=EPS)

    sum_vols = vols1[:, :, None] + vols2[:, None, :]

    # filter malformed boxes
    good_boxes = (enclosing_vols > 2 * EPS) * (sum_vols > 4 * EPS)

    if rotated_boxes:
        inter_areas = torch.zeros((B, K1, K2), dtype=torch.float32)
        rect1 = rect1.cpu()
        rect2 = rect2.cpu()
        nums_k2_np = to_list_1d(nums_k2)
        non_rot_inter_areas_np = to_list_3d(non_rot_inter_areas)
        for b in range(B):
            for k1 in range(K1):
                for k2 in range(K2):
                    if nums_k2 is not None and k2 >= nums_k2_np[b]:
                        break
                    if non_rot_inter_areas_np[b][k1][k2] == 0:
                        continue
                    ##### compute volume of intersection
                    inter = polygon_clip_unnest(rect1[b, k1], rect2[b, k2])
                    if len(inter) > 0:
                        xs = torch.stack([x[0] for x in inter])
                        ys = torch.stack([x[1] for x in inter])
                        inter_areas[b, k1, k2] = torch.abs(
                            torch.dot(xs, torch.roll(ys, 1))
                            - torch.dot(ys, torch.roll(xs, 1))
                        )
        inter_areas.mul_(0.5)
    else:
        inter_areas = non_rot_inter_areas

    inter_areas = inter_areas.to(corners1.device)
    ### gIOU = iou - (1 - sum_vols/enclose_vol)
    inter_vols = inter_areas * height
    if return_inter_vols_only:
        return inter_vols

    union_vols = (sum_vols - inter_vols).clamp(min=EPS)
    ious = inter_vols / union_vols
    giou_second_term = -(1 - union_vols / enclosing_vols)
    gious = ious + giou_second_term
    gious *= good_boxes
    if nums_k2 is not None:
        mask = torch.zeros((B, K1, K2), device=height.device, dtype=torch.float32)
        for b in range(B):
            mask[b, :, : nums_k2[b]] = 1
        gious *= mask
    return gious

generalized_box3d_iou_tensor_jit = torch.jit.script(generalized_box3d_iou_tensor)

"""

def generalized_box3d_iou(
    corners1: torch.Tensor,
    corners2: torch.Tensor,
    nums_k2: torch.Tensor,
    rotated_boxes: bool = True,
    return_inter_vols_only: bool = False,
    needs_grad: bool = False,
):
    # May need to remove the needs_grad parameter so GIoU is calculated with cython and not tensor_jit
    if needs_grad is True or box_intersection is None:
        context = torch.enable_grad if needs_grad else torch.no_grad
        with context():
            return None
            #return generalized_box3d_iou_tensor_jit(
                #corners1, corners2, nums_k2, rotated_boxes, return_inter_vols_only
            #)

    else:
        # Cythonized implementation of GIoU
        with torch.no_grad():
            return generalized_box3d_iou_cython(
                corners1, corners2, nums_k2, rotated_boxes, return_inter_vols_only
            )

def flip_axis_to_camera_tensor(pc):
    """Flip X-right, Y-forward, Z-up to X-right, Y-down, Z-forward."""
    pc2 = torch.clone(pc)
    # cam X,Y,Z = depth X,-Z,Y
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]
    pc2[..., 1] *= -1
    return pc2


def roty_batch_tensor(angle):
    """
    Compute rotation matrices around the Y-axis (yaw).
    
    Args:
        angle (Tensor): Yaw angles in radians (N,).
    
    Returns:
        Tensor: Rotation matrices (N, 3, 3).
    """
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    
    # Create the rotation matrix
    R = torch.zeros((angle.shape[0], 3, 3), device=angle.device)
    R[:, 0, 0] = cos_theta
    R[:, 0, 2] = sin_theta
    R[:, 1, 1] = 1.0
    R[:, 2, 0] = -sin_theta
    R[:, 2, 2] = cos_theta
    
    return R

def get_3d_box_batch_tensor(box_size, angle, center):
    """
    Generate 3D bounding box corners based on size, rotation, and center.
    
    Args:
        box_size (Tensor): Size of the bounding box (N, 3) [length, height, width].
        angle (Tensor): Rotation angle in radians (N,).
        center (Tensor): Center coordinates of the bounding box (N, 3).
    
    Returns:
        Tensor: 3D corner coordinates (N, 8, 3).
    """
    assert box_size.ndim == 2 and angle.ndim == 1 and center.ndim == 2

    # Compute rotation matrices for the angles
    R = roty_batch_tensor(angle)

    # Bounding box dimensions
    l = torch.unsqueeze(box_size[..., 0], -1)
    w = torch.unsqueeze(box_size[..., 1], -1)
    h = torch.unsqueeze(box_size[..., 2], -1)

    # Predefine 8 corners in the local coordinate system
    corners_3d = torch.zeros((box_size.shape[0], 8, 3), device=box_size.device)
    corners_3d[:, :, 0] = torch.cat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=-1)
    corners_3d[:, :, 1] = torch.cat([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dim=-1)
    corners_3d[:, :, 2] = torch.cat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=-1)

    # Apply rotation and translation to the corners
    corners_3d = torch.einsum("bij,bkj->bki", R, corners_3d) 
    corners_3d += center.unsqueeze(1)

    return corners_3d
