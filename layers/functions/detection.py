#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch

from ..bbox_utils import decode, nms
from torch.autograd import Function


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
            loc_data: (tensor) Location predictions
                Shape: [batch, num_priors*4]
            conf_data: (tensor) Confidence predictions
                Shape: [batch*num_priors, num_classes]
            prior_data: (tensor) Prior boxes
                Shape: [1, num_priors, 4]

        Returns:
            output: (tensor) Final detected objects
                Shape: [batch, num_classes, top_k, 5] (score, x1, y1, x2, y2)
        """
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        # print(loc_data.view(-1, 4).device)
        # print(batch_priors.device)

        decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, self.num_classes, self.top_k, 5, device=loc_data.device)

        for i in range(num):  # Batch iteration
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):  # Skip background class
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # Confidence filter
                scores = conf_scores[cl][c_mask]

                if scores.numel() == 0:  # No detections for this class
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)

                ids, count = nms(boxes_, scores, self.nms_thresh, self.nms_top_k)
                count = min(count, self.top_k)

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes_[ids[:count]]), 1)

        return output