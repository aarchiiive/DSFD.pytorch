#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch

from ..bbox_utils import decode, nms
from torch.autograd import Function
from torchvision.ops import batched_nms


# class Detect(Function):
#     """At test time, Detect is the final layer of SSD.  Decode location preds,
#     apply non-maximum suppression to location predictions based on conf
#     scores and threshold to a top_k number of output predictions for both
#     confidence score and locations.
#     """

#     def __init__(self, cfg):
#         self.num_classes = cfg.NUM_CLASSES
#         self.top_k = cfg.TOP_K
#         self.nms_thresh = cfg.NMS_THRESH
#         self.conf_thresh = cfg.CONF_THRESH
#         self.variance = cfg.VARIANCE
#         self.nms_top_k = cfg.NMS_TOP_K

#     def forward(self, loc_data, conf_data, prior_data):
#         """
#         Args:
#             loc_data: (tensor) Loc preds from loc layers
#                 Shape: [batch,num_priors*4]
#             conf_data: (tensor) Shape: Conf preds from conf layers
#                 Shape: [batch*num_priors,num_classes]
#             prior_data: (tensor) Prior boxes and variances from priorbox layers
#                 Shape: [1,num_priors,4]
#         """
#         num = loc_data.size(0)
#         num_priors = prior_data.size(0)

#         conf_preds = conf_data.view(
#             num, num_priors, self.num_classes).transpose(2, 1)
#         batch_priors = prior_data.view(-1, num_priors,
#                                        4).expand(num, num_priors, 4)
#         batch_priors = batch_priors.contiguous().view(-1, 4)

#         decoded_boxes = decode(loc_data.view(-1, 4),
#                                batch_priors, self.variance)
#         decoded_boxes = decoded_boxes.view(num, num_priors, 4)

#         output = torch.zeros(num, self.num_classes, self.top_k, 5)

#         for i in range(num):
#             boxes = decoded_boxes[i].clone()
#             conf_scores = conf_preds[i].clone()

#             for cl in range(1, self.num_classes):
#                 c_mask = conf_scores[cl].gt(self.conf_thresh)
#                 scores = conf_scores[cl][c_mask]

#                 if scores.dim() == 0:
#                     continue
#                 l_mask = c_mask.unsqueeze(1).expand_as(boxes)
#                 boxes_ = boxes[l_mask].view(-1, 4)
#                 ids, count = nms(
#                     boxes_, scores, self.nms_thresh, self.nms_top_k)
#                 count = count if count < self.top_k else self.top_k

#                 output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
#                                                    boxes_[ids[:count]]), 1)

#         return output

# class Detect:
#     """At test time, Detect is the final layer of SSD.
#     Decode location predictions, apply non-maximum suppression
#     based on confidence scores, and return top_k predictions.
#     """

#     def __init__(self, num_classes, top_k, nms_thresh, conf_thresh, variance, nms_top_k):
#         """
#         Args:
#             num_classes (int): Number of classes
#             top_k (int): Maximum number of predictions to keep
#             nms_thresh (float): Non-maximum suppression threshold
#             conf_thresh (float): Confidence threshold for filtering detections
#             variance (list): Variance for bounding box decoding
#             nms_top_k (int): Max number of detections after NMS
#         """
#         self.num_classes = num_classes
#         self.top_k = top_k
#         self.nms_thresh = nms_thresh
#         self.conf_thresh = conf_thresh
#         self.variance = variance
#         self.nms_top_k = nms_top_k

#     def __call__(self, loc_data, conf_data, prior_data):
#         """
#         Args:
#             loc_data: (tensor) Location predictions
#                 Shape: [batch, num_priors*4]
#             conf_data: (tensor) Confidence predictions
#                 Shape: [batch*num_priors, num_classes]
#             prior_data: (tensor) Prior boxes
#                 Shape: [1, num_priors, 4]

#         Returns:
#             output: (tensor) Final detected objects
#                 Shape: [batch, num_classes, top_k, 5] (score, x1, y1, x2, y2)
#         """
#         num = loc_data.size(0)
#         num_priors = prior_data.size(0)

#         conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
#         batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
#         batch_priors = batch_priors.contiguous().view(-1, 4)

#         # print(loc_data.view(-1, 4).device)
#         # print(batch_priors.device)

#         decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, self.variance)
#         decoded_boxes = decoded_boxes.view(num, num_priors, 4)

#         output = torch.zeros(num, self.num_classes, self.top_k, 5, device=loc_data.device)

#         for i in range(num):  # Batch iteration
#             boxes = decoded_boxes[i].clone()
#             conf_scores = conf_preds[i].clone()

#             for cl in range(1, self.num_classes):  # Skip background class
#                 c_mask = conf_scores[cl].gt(self.conf_thresh)  # Confidence filter
#                 scores = conf_scores[cl][c_mask]

#                 if scores.numel() == 0:  # No detections for this class
#                     continue

#                 l_mask = c_mask.unsqueeze(1).expand_as(boxes)
#                 boxes_ = boxes[l_mask].view(-1, 4)

#                 ids, count = nms(boxes_, scores, self.nms_thresh, self.nms_top_k)
#                 count = min(count, self.top_k)

#                 output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes_[ids[:count]]), 1)

#         return output


class Detect:
    """At test time, Detect is the final layer of SSD.
    Decode location predictions, apply non-maximum suppression
    based on confidence scores, and return top_k predictions.
    """

    def __init__(self, num_classes, top_k, nms_thresh, conf_thresh, variance, nms_top_k):
        """
        Args:
            num_classes (int): Number of classes
            top_k (int): Maximum number of predictions to keep
            nms_thresh (float): Non-maximum suppression threshold
            conf_thresh (float): Confidence threshold for filtering detections
            variance (list): Variance for bounding box decoding
            nms_top_k (int): Max number of detections after NMS
        """
        self.num_classes = num_classes
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = variance
        self.nms_top_k = nms_top_k

    def __call__(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data (Tensor): Location predictions of shape [batch, num_priors*4]
            conf_data (Tensor): Confidence predictions of shape [batch*num_priors, num_classes]
            prior_data (Tensor): Prior boxes of shape [1, num_priors, 4]

        Returns:
            output (Tensor): Final detected objects of shape [batch, num_classes, top_k, 5]
                            (each detection: [score, x1, y1, x2, y2])
        """
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        # Reshape confidence predictions to [batch, num_classes, num_priors]
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        # Expand prior boxes for the entire batch and reshape to [batch*num_priors, 4]
        batch_priors = prior_data.view(1, num_priors, 4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        # Decode location predictions using prior boxes and variances
        decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        # Prepare output tensor of shape [batch, num_classes, top_k, 5]
        output = torch.zeros(num, self.num_classes, self.top_k, 5, device=loc_data.device)

        # Process each image in the batch
        for i in range(num):
            boxes = decoded_boxes[i]       # shape: [num_priors, 4]
            conf_scores = conf_preds[i]      # shape: [num_classes, num_priors]

            detection_list = []
            # Process each class (skip background, assumed to be class 0)
            for cl in range(1, self.num_classes):
                mask = conf_scores[cl] > self.conf_thresh
                if mask.sum() == 0:
                    continue

                scores = conf_scores[cl][mask]
                boxes_cl = boxes[mask]
                # Limit per-class detections to nms_top_k (to mimic original behavior)
                if scores.numel() > self.nms_top_k:
                    sorted_inds = torch.argsort(scores, descending=True)
                    top_inds = sorted_inds[:self.nms_top_k]
                    scores = scores[top_inds]
                    boxes_cl = boxes_cl[top_inds]

                # Create a tensor for labels (all elements are current class)
                labels = torch.full((scores.size(0),), cl, dtype=torch.int64, device=boxes.device)
                detection_list.append((boxes_cl, scores, labels))

            if len(detection_list) == 0:
                continue

            # Concatenate detections from all classes
            all_boxes = torch.cat([x[0] for x in detection_list], dim=0)
            all_scores = torch.cat([x[1] for x in detection_list], dim=0)
            all_labels = torch.cat([x[2] for x in detection_list], dim=0)

            # Apply batched NMS across all classes simultaneously
            keep = batched_nms(all_boxes, all_scores, all_labels, self.nms_thresh)
            # 'keep' is sorted in descending order of scores

            # Prepare an output tensor for the current image [num_classes, top_k, 5]
            image_output = torch.zeros(self.num_classes, self.top_k, 5, device=loc_data.device)

            # Separate NMS results back into each class
            for cl in range(1, self.num_classes):
                cls_mask = all_labels[keep] == cl
                if cls_mask.sum() == 0:
                    continue
                cls_keep = keep[cls_mask]
                cls_scores = all_scores[cls_keep]
                cls_boxes = all_boxes[cls_keep]
                # Limit to top_k detections per class
                if cls_scores.numel() > self.top_k:
                    sorted_inds = torch.argsort(cls_scores, descending=True)[:self.top_k]
                    cls_scores = cls_scores[sorted_inds]
                    cls_boxes = cls_boxes[sorted_inds]
                num_det = cls_scores.numel()
                if num_det > 0:
                    # First column is score, followed by box coordinates
                    image_output[cl, :num_det, 0] = cls_scores
                    image_output[cl, :num_det, 1:] = cls_boxes

            output[i] = image_output

        return output