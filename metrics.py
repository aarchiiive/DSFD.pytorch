from typing import Tuple, Optional

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def norm_score(org_pred_list):
    norm_pred_list = []
    max_score = torch.finfo(torch.float32).min
    min_score = torch.finfo(torch.float32).max

    for i in range(len(org_pred_list)):
        pred = org_pred_list[i]
        scores = pred[..., 4]
        max_score = max(max_score, scores.max())
        min_score = min(min_score, scores.min())

    for i in range(len(org_pred_list)):
        pred = org_pred_list[i]
        scores = pred[..., 4]
        if max_score != min_score:
            norm_scores = (scores - min_score) / (max_score - min_score)
        else:
            norm_scores = (scores - (min_score - 1))
        org_pred_list[i][..., 4] = norm_scores

    norm_pred_list = org_pred_list
    return norm_pred_list


def compute_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: shape (N, 4) -> [[x1, y1, x2, y2],
                        [x1, y1, x2, y2],
                         ...
                       ]
    b: shape (4,)   -> [x1, y1, x2, y2]

    Returns:
        A 1D Tensor of size N, each element representing
        the IOU between one of the boxes in 'a' and 'b'.
    """
    # Calculate the top-left corner of the intersection area
    x1 = torch.max(a[:, 0], b[0])
    y1 = torch.max(a[:, 1], b[1])

    # Calculate the bottom-right corner of the intersection area
    x2 = torch.min(a[:, 2], b[2])
    y2 = torch.min(a[:, 3], b[3])

    # Compute width and height of the intersection area (+1 to match pixel-based bounding box convention)
    w = x2 - x1 + 1
    h = y2 - y1 + 1

    # Compute the area of the intersection
    inter = w * h

    # Compute the area of each box in 'a' and the box 'b' (+1 applied similarly)
    aarea = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

    # Calculate IOU = inter / (aarea + barea - inter)
    iou = inter / (aarea + barea - inter)

    # If intersection dimensions become zero or negative, set IOU to 0
    iou[w <= 0] = 0
    iou[h <= 0] = 0

    return iou


def eval_image(preds: torch.Tensor, targets: torch.Tensor, iou_thresh: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the predictions for a single image by computing cumulative recall counts.

    Args:
        preds (torch.Tensor): Tensor containing predicted bounding boxes and scores.
                              Expected shape: [num_preds, ...], where the first 4 elements in each row represent the bounding box.
        targets (torch.Tensor): Tensor containing ground truth bounding boxes.
                                Expected shape: [num_targets, ...].
        iou_thresh (float): IOU threshold for determining a correct detection.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - pred_recall (torch.Tensor): Tensor of shape [num_preds, 1] representing the cumulative number of true detections
                                          for each prediction.
            - proposal_list (torch.Tensor): Tensor of shape [num_preds, 1] representing proposals (currently set to all ones).
    """
    # Initialize tensor to store cumulative recall counts for each prediction (Type: torch.Tensor)
    pred_recall = torch.zeros((len(preds), 1))

    # Initialize tensor to mark whether a target has been detected (Type: torch.Tensor)
    recall_list = torch.zeros((len(targets), 1))

    # Initialize tensor for proposals (for this implementation, all proposal values are 1) (Type: torch.Tensor)
    proposal_list = torch.ones((len(preds), 1))

    # Iterate through each predicted bounding box
    for i in range(len(preds)):
        # Compute the IoU between the current predicted bounding box and all target boxes.
        # Here, preds[i, :4] extracts the bounding box coordinates from the i-th prediction.
        overlap_list = compute_iou(targets, preds[i, :4])

        # Find the maximum IoU value and its corresponding index among the targets.
        max_overlap, max_idx = overlap_list.max(dim=0)

        # If the maximum IoU exceeds the threshold, mark the corresponding target as detected.
        if max_overlap >= iou_thresh:
            recall_list[max_idx] = 1

        # Retrieve the indices of targets that have been detected (where recall_list equals 1)
        r_keep_index = torch.where(recall_list == 1)[0]

        # Update the cumulative recall count for the current prediction.
        pred_recall[i] = len(r_keep_index)

    return pred_recall, proposal_list


def image_pr_info(num_thresh: int,
                  preds: torch.Tensor,
                  proposal_list: torch.Tensor,
                  pred_recall: torch.Tensor) -> torch.Tensor:
    """
    Compute image precision-recall information for a range of thresholds.

    Args:
        num_thresh (int): Number of thresholds.
        preds (torch.Tensor): Tensor containing prediction information.
                              Assumes at least 5 columns, where the 5th column (index 4) holds scores.
        proposal_list (torch.Tensor): Tensor indicating proposals (e.g., binary indicators).
        pred_recall (torch.Tensor): Tensor of cumulative recall values.

    Returns:
        torch.Tensor: A tensor of shape [num_thresh, 2] where:
                      - Column 0 stores the number of proposals.
                      - Column 1 stores the recall value corresponding to the threshold.
    """
    # Initialize tensor to store precision-recall info for each threshold.
    img_pr_info = torch.zeros((num_thresh, 2))

    # Loop over each threshold index.
    for t in range(num_thresh):
        # Calculate the threshold value.
        thresh = 1 - t / num_thresh

        # Create a boolean mask where predictions' score (column index 4) is above the threshold.
        mask = preds[:, 4] >= thresh

        # Get the indices of predictions that meet the threshold criterion.
        indices = torch.nonzero(mask, as_tuple=True)[0]

        # If there are any valid indices, take the last one as the reference index; otherwise, set to None.
        r_index = indices[-1].item() if indices.numel() > 0 else None

        if r_index is None:
            # No predictions meet the threshold; set both proposal count and recall to zero.
            img_pr_info[t, 0] = 0.0
            img_pr_info[t, 1] = 0.0
        else:
            # For predictions up to the reference index, find where proposals are indicated (value == 1).
            p_index = torch.nonzero(proposal_list[:r_index+1] == 1, as_tuple=True)[0]

            # Store the number of proposals (count of non-zero entries) in the first column.
            img_pr_info[t, 0] = p_index.numel()

            # Store the corresponding recall value (from pred_recall) in the second column.
            img_pr_info[t, 1] = pred_recall[r_index].item()

    return img_pr_info


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """
    Compute VOC AP given recall and precision arrays.

    Args:
        rec (np.ndarray): Array of recall values.
        prec (np.ndarray): Array of precision values.

    Returns:
        float: Average Precision (AP) score.
    """
    # Append boundary values to recall and precision
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # Make the precision envelope (ensure precision is non-increasing)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Identify indices where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1

    # Sum (\Delta recall) * precision at those indices to get AP
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
    return ap


def save_pr_curve(pr_curve: torch.Tensor, save_path: str) -> None:
    """
    Save the precision-recall curve to a text file.

    Args:
        pr_curve (torch.Tensor): Tensor containing precision-recall values.
        save_path (str): Path to save the precision-recall curve.
    """
    if isinstance(pr_curve, torch.Tensor):
        pr_curve_np = pr_curve.detach().cpu().numpy() if pr_curve.is_cuda else pr_curve.numpy()
    elif isinstance(pr_curve, np.ndarray):
        pr_curve_np = pr_curve

    precision = pr_curve_np[:, 0]
    recall = pr_curve_np[:, 1]

    # Compute AP
    ap = voc_ap(recall, precision)

    print(f"Average Precision: {ap}")

    df = pd.DataFrame({'Recall': recall, 'Precision': precision})

    sns.set(style='whitegrid')
    sns.set_palette("bright")
    plt.figure(figsize=(8, 6))
    sns.lineplot(x='Recall', y='Precision', data=df)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()
    plt.savefig(save_path)
    plt.close()

    return ap