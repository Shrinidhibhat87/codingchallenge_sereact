"""
Utility file that contains the IoU evaluator class.
"""

import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.affinity import scale


class IoUEvaluator:
    """
    Custom evaluator for measuring the quality of bounding box predictions
    based on IoU (Intersection-over-Union).

    Attributes:
        iou_thresholds (list): List of IoU thresholds for evaluation (e.g., [0.25, 0.5]).
        iou_scores (list): Stores IoU scores for all evaluated boxes.
        threshold_hits (dict): Tracks the number of predictions meeting each IoU threshold.
        total_boxes (int): Total number of ground truth boxes evaluated.
    """

    def __init__(self, iou_thresholds=[0.25, 0.5]):
        """
        Initialize the IoUEvaluator with specified IoU thresholds.

        Args:
            iou_thresholds (list): IoU thresholds for evaluating predictions.
        """
        self.iou_thresholds = iou_thresholds
        self.reset()

    def reset(self):
        """Reset the evaluator's metrics for a new evaluation run."""
        self.iou_scores = []
        self.threshold_hits = {t: 0 for t in self.iou_thresholds}
        self.total_boxes = 0

    @staticmethod
    def calculate_iou(box_a, box_b):
        """
        Compute IoU between two 3D bounding boxes.
        Static method does not take any self, cls variable.
        Essentially, it is a utility function that can be moved elsewhere.

        Args:
            box_a (ndarray): Predicted bounding box of shape (8, 3).
            box_b (ndarray): Ground truth bounding box of shape (8, 3).

        Returns:
            float: IoU value between the two boxes.
        """
        # Use of external libraries to calculate the IoU (currently shapely).
        # There is a possibility to make use of more complicated, but accurate IoU calc.
        # https://github.com/facebookresearch/pytorch3d
        # https://pytorch3d.org/docs/iou3d

        # Using shapely, we are converting 3D bounding boxes to 2D polygons after projections.
        # Therefore we assume here that assumes z-plane is dominant
        # projecting the predicted bounding box onto the x-y plane
        # predicted_polygon = Polygon(box_a[:, :2])
        # ground_truth_polygon = Polygon(box_b[:, :2])
        
        # Project to 2D and compute convex hull for each box
        def get_convex_hull(box):
            points = box[:, :2]
            multipoint = MultiPoint(points)
            hull = multipoint.convex_hull
            
            return hull
        
        # Get the convex hull for each box
        hull_a = get_convex_hull(box_a)
        hull_b = get_convex_hull(box_b)

        # Calculate the IoU
        intersection = hull_a.intersection(hull_b).area
        union = hull_a.union(hull_b).area

        # Edge case if there is no union
        if union == 0:
            return 0.0

        return intersection / union


    def update(self, pred_boxes, gt_boxes):
        """
        Update the evaluator with a batch of predicted and ground truth boxes.

        Args:
            pred_boxes (list of ndarray): List of predicted boxes for the batch.
            gt_boxes (list of ndarray): List of ground truth boxes for the batch.
        """
        for pred_box, gt_box in zip(pred_boxes, gt_boxes):
            iou = self.calculate_iou(pred_box, gt_box)
            self.iou_scores.append(iou)

            # Check if the IoU meets each threshold.
            for thresh in self.iou_thresholds:
                if iou >= thresh:
                    self.threshold_hits[thresh] += 1

            # Track total ground truth boxes processed.
            self.total_boxes += 1

    def compute_metrics(self):
        """
        Compute the final metrics after all updates.

        Returns:
            dict: A dictionary containing mean IoU and threshold-based accuracy.
        """
        mean_iou = sum(self.iou_scores) / len(self.iou_scores) if self.iou_scores else 0
        threshold_accuracy = {
            thresh: hits / self.total_boxes if self.total_boxes > 0 else 0
            for thresh, hits in self.threshold_hits.items()
        }

        return {"mean_iou": mean_iou, "threshold_accuracy": threshold_accuracy}

    def __str__(self):
        """Generate a string summary of the evaluator's metrics."""
        metrics = self.compute_metrics()
        thresh_str = ", ".join(
            [f"IoU@{t}: {metrics['threshold_accuracy'][t]:.2f}" for t in self.iou_thresholds]
        )

        return f"Mean IoU: {metrics['mean_iou']:.2f}, {thresh_str}"
