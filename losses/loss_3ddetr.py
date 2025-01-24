"""
The file conatins the loss function used to match the 3D bounding box.
This file is also inspired from the official implementation of 3DDETR.
Link: https://facebookresearch.github.io/3detr
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

# Some util functions that needs to be imported as well

from scipy.optimize import linear_sum_assignment
from utils.bounding_box_operations import generalized_box3d_iou

def huber_loss(error, delta=1.0):
    """
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss


def is_distributed():
    if not dist.is_available() or not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()

def all_reduce_sum(tensor):
    if not is_distributed():
        return tensor
    dim_squeeze = False
    if tensor.ndim == 0:
        tensor = tensor[None, ...]
        dim_squeeze = True
    torch.distributed.all_reduce(tensor)
    if dim_squeeze:
        tensor = tensor.squeeze(0)
    return tensor


def all_reduce_average(tensor):
    val = all_reduce_sum(tensor)
    return val / get_world_size()

class Matcher_loss(nn.Module):
    def __init__(
        self,
        cost_objectness,
        cost_giou,
        cost_center
    ):
        super().__init__()
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center
    
    @torch.no_grad()
    def forward(
        self,
        outputs,
        targets
    ):
        # Get the batch size
        batch_size = outputs["objectness_prob"].shape[0]
        # Get the number of queries
        num_queries = outputs["objectness_prob"].shape[1]
        
        # Objectness cost (batch, num_queries, 1). Negative log prob of objectness
        objectness_matching = -outputs["objectness_prob"].unsqueeze(-1)
        
        # Center cost (batch, num_queries, ngt). Distance b/w predicted and ground truth
        center_matching = outputs['center_dist'].detach()
        
        # GIoU cost (batch, num_queries, ngt). Negative GIoU score.
        giou_matching = -outputs["gious"].detach()
        
        # Calculate the final cost
        final_cost = (
            self.cost_objectness * objectness_matching +
            self.cost_center * center_matching +
            self.cost_giou * giou_matching
        )
        
        # Detach from GPU to CPU
        final_cost = final_cost.detach().cpu().numpy()
        
        # Append the assignments
        assignments = []
        
        # Auxiliary variables useful for batched loss computation
        # If batch_size == batchsize and nprop == num_queries, remove the lines below.
        batchsize, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batchsize, nprop], dtype=torch.int64, device=outputs["semantic_class_prob"].device
        )
        proposal_matched_mask = torch.zeros(
            [batchsize, nprop], dtype=torch.float32, device=outputs["semantic_class_prob"].device
        )
        
        # Here, we perform the Hungarian matching
        for b in range(batch_size):
            assign = []
            if "nactual_gt" in targets and targets["nactual_gt"][b] > 0:
                nactual_gt = targets["nactual_gt"][b]
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=outputs["semantic_class_prob"].device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


class SetCriterion(nn.Module):
    def __init__(
        self,
        matcher_loss,
        loss_weight_dict
    ):
        super().__init__()
        self.matcher_loss = matcher_loss
        self.loss_weight_dict = loss_weight_dict

        # Losses
        self.loss_functions = {
            "loss_angle": self.loss_angle,
            "loss_center": self.loss_center,
            "loss_size": self.loss_size,
            "loss_giou": self.loss_giou,
        }

    def loss_angle(
        self,
        outputs,
        targets,
        assignments
    ):
        # Extract angle logits and residuals from the outputs
        angle_logits = outputs["angle_logits"]
        angle_residual = outputs["angle_residual_normalized"]

        # Check if there are any ground truth boxes
        if targets["num_boxes_replica"] > 0:
            # Extract ground truth angle labels and residuals
            gt_angle_label = targets["gt_angle_class_label"]
            gt_angle_residual = targets["gt_angle_residual_label"]
            # Normalize the ground truth angle residuals
            gt_angle_residual_normalized = gt_angle_residual / (np.pi / 12)
            # Gather the ground truth angle labels based on the assignments
            gt_angle_label = torch.gather(
                gt_angle_label, 1, assignments["per_prop_gt_inds"]
            )
            # Compute the cross-entropy loss for angle classification
            angle_cls_loss = F.cross_entropy(
                angle_logits.transpose(2, 1), gt_angle_label, reduction="none"
            )
            # Mask the classification loss with the proposal matched mask and sum it
            angle_cls_loss = (
                angle_cls_loss * assignments["proposal_matched_mask"]
            ).sum()

            # Gather the normalized ground truth angle residuals based on the assignments
            gt_angle_residual_normalized = torch.gather(
                gt_angle_residual_normalized, 1, assignments["per_prop_gt_inds"]
            )
            # Create a one-hot encoding of the ground truth angle labels
            gt_angle_label_one_hot = torch.zeros_like(
                angle_residual, dtype=torch.float32
            )
            gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)

            # Extract the angle residuals for the ground truth classes
            angle_residual_for_gt_class = torch.sum(
                angle_residual * gt_angle_label_one_hot, -1
            )
            # Compute the Huber loss for angle regression
            angle_reg_loss = huber_loss(
                angle_residual_for_gt_class - gt_angle_residual_normalized, delta=1.0
            )
            # Mask the regression loss with the proposal matched mask and sum it
            angle_reg_loss = (
                angle_reg_loss * assignments["proposal_matched_mask"]
            ).sum()

            # Normalize the classification and regression losses by the number of boxes
            angle_cls_loss /= targets["num_boxes"]
            angle_reg_loss /= targets["num_boxes"]
        else:
            # If there are no ground truth boxes, set the losses to zero
            angle_cls_loss = torch.zeros(1, device=angle_logits.device).squeeze()
            angle_reg_loss = torch.zeros(1, device=angle_logits.device).squeeze()
        
        # Return the angle classification and regression losses
        return {"loss_angle_cls": angle_cls_loss, "loss_angle_reg": angle_reg_loss}
    
    def loss_center(
        self,
        outputs,
        targets,
        assignments
    ):
        # Get the distance of the centers from the outputs
        center_dist = outputs["center_dist"]
        
        # Check if there are any ground truth boxes
        if targets["num_boxes_replica"] > 0:
            # Select appropriate distances by using proposal to ground truth matching
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # Zero-out non-matched proposals
            center_loss = center_loss * assignments["proposal_matched_mask"]
            # Sum the center loss
            center_loss = center_loss.sum()

            # Normalize the center loss by the number of boxes
            if targets["num_boxes"] > 0:
                center_loss /= targets["num_boxes"]
        else:
            # If there are no ground truth boxes, set the center loss to zero
            center_loss = torch.zeros(1, device=center_dist.device).squeeze()

        # Return the center loss
        return {"loss_center": center_loss}
    
    def loss_giou(
        self,
        outputs,
        targets,
        assignments
    ):
        # Get the GIoU distances
        gious_dist = 1 - outputs["gious"]
        
        # Select appropriate GIoUs by using proposal to ground truth matching
        giou_loss = torch.gather(
            gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
        ).squeeze(-1)
        # Zero-out non-matched proposals
        giou_loss = giou_loss * assignments["proposal_matched_mask"]
        giou_loss = giou_loss.sum()

        # Normalize the GIoU loss by the number of boxes
        if targets["num_boxes"] > 0:
            giou_loss /= targets["num_boxes"]

        # Return the GIoU loss
        return {"loss_giou": giou_loss}
    
    def loss_size(
        self,
        outputs,
        targets,
        assignments
    ):
        # Get the ground truth bbox sizes
        gt_box_sizes = targets["gt_box_sizes_normalized"]
        # get the predicted bbox sizes
        pred_box_sizes = outputs["size_normalized"]
        
        # Check if there are any ground truth boxes
        if targets["num_boxes_replica"] > 0:
            # Construct the ground truth boxes as (batch, nprop, 3) by using proposal to ground truth matching
            gt_box_sizes = torch.stack(
                [
                    torch.gather(
                        gt_box_sizes[:, :, i], 1, assignments["per_prop_gt_inds"]
                    )
                    for i in range(gt_box_sizes.shape[-1])
                ],
                dim=-1,
            )
            # Get the loss
            size_loss = F.l1_loss(pred_box_sizes, gt_box_sizes, reduction="none").sum(
                dim=-1
            )
            
            # Zero-out non-matched proposals
            size_loss *= assignments["proposal_matched_mask"]
            size_loss = size_loss.sum()
            
            size_loss /= targets["num_boxes"]
        
        else:
            size_loss = torch.zeros(1, device=pred_box_sizes.device).squeeze()
        
        # Return the Size loss
        return {"loss_size": size_loss}
    
    def single_output_forward(
        self,
        outputs,
        targets  
    ):
        # Compute the Generalized Intersection over Union (GIoU) between predicted and ground truth boxes
        gious = generalized_box3d_iou(
            outputs["box_corners"],
            targets["gt_box_corners"],
            targets["nactual_gt"],
            rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
            needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0), # Check if we really need this parameter.
        )

        # Store the GIoU in the outputs dictionary
        outputs["gious"] = gious

        # Compute the L1 distance between predicted and ground truth box centers
        center_dist = torch.cdist(
            outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
        )
        # Store the center distances in the outputs dictionary
        outputs["center_dist"] = center_dist

        # Perform the matching between predictions and ground truth boxes
        assignments = self.matcher_loss(outputs, targets)
        
        # Initialize a dictionary to store the losses
        losses = {}
        
        # Iterate over each loss function
        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            # Check if the loss weight is greater than 0 or if the loss weight key is not in the dictionary
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                # Compute the current loss
                curr_loss = self.loss_functions[k](outputs, targets, assignments)
                # Update the losses dictionary with the current loss
                losses.update(curr_loss)

        # Initialize the final loss to 0
        final_loss = 0
        # Iterate over each loss weight in the dictionary
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                # Multiply the loss by its weight
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                # Add the weighted loss to the final loss
                final_loss += losses[k.replace("_weight", "")]
        
        # Return the final loss and the individual losses
        return final_loss, losses
    
    def forward(self, outputs, targets):
        # Calculate the number of ground truth boxes present in each batch
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()
        
        # Calculate the total number of boxes across all replicas and clamp it to a minimum of 1
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()
        
        # Update the targets dictionary with the number of actual ground truth boxes
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        
        # Update the targets dictionary with the number of boxes on this worker for distributed training
        targets["num_boxes_replica"] = nactual_gt.sum().item()

        # Compute the loss and loss dictionary for the main outputs
        loss, loss_dict = self.single_output_forward(outputs["outputs"], targets)

        # If there are auxiliary outputs, compute the loss for each of them
        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                # Compute the intermediate loss and loss dictionary for the auxiliary outputs
                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets
                )

                # Accumulate the intermediate loss to the total loss
                loss += interm_loss
                
                # Update the loss dictionary with the intermediate losses
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        
        # Return the total loss and the loss dictionary
        return loss, loss_dict
        

def build_loss_object(args):
    """
    Build the loss object based on the arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        nn.Module: The loss object.
    """
    # There are 4 mjaor components of the matching loss
    matcher_loss = Matcher_loss(
        cost_giou=args.matcher_cost_giou,
        cost_center=args.matcher_cost_center,
        cost_objectness=args.matcher_cost_objectness
    )
    
    loss_weight_dict = {
        "loss_giou_weight": args.loss_giou_weight,
        "loss_center_weight": args.loss_center_weight,
        "loss_angle_cls_weight": args.loss_angle_cls_weight,
        "loss_angle_reg_weight": args.loss_angle_reg_weight,
        "loss_size_weight": args.loss_size_weight,
        "loss_no_object_weight": args.loss_no_object_weight
    }
    
    criterion = SetCriterion(
        matcher_loss=matcher_loss,
        loss_weight_dict=loss_weight_dict
    )
    
    return criterion
