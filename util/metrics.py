"""Evaluation metrics for segmentation."""

import numpy as np
import torch


def dice_score(pred: np.ndarray,
               target: np.ndarray,
               smooth: float = 1e-5) -> float:
    """
    Dice Similarity Coefficient.

    Args:
        pred, target: binary numpy arrays of any shape.
    Returns:
        DSC ∈ [0, 1]
    """
    intersection = (pred.astype(bool) & target.astype(bool)).sum()
    total = pred.astype(bool).sum() + target.astype(bool).sum()
    return float((2.0 * intersection + smooth) / (total + smooth))


def dice_score_batch(pred_batch: torch.Tensor,
                     target_batch: torch.Tensor) -> list:
    """
    Compute Dice for every item in a batch.

    Args:
        pred_batch:   (B, H, W) binary tensor
        target_batch: (B, H, W) binary tensor

    Returns:
        list of B float DSC values
    """
    dices = []
    for pred, target in zip(pred_batch, target_batch):
        p = pred.cpu().numpy().astype(bool)
        t = target.cpu().numpy().astype(bool)
        dices.append(dice_score(p, t))
    return dices


def hausdorff_95(pred: np.ndarray,
                 target: np.ndarray,
                 spacing: tuple = (1.0, 1.0)) -> float:
    """
    95th-percentile Hausdorff Distance (HD95).

    Lower is better. Returns float('nan') if either mask is empty.

    Args:
        pred, target: binary numpy arrays (H, W)
        spacing:      pixel/voxel spacing in mm
    """
    from scipy.ndimage import binary_erosion
    from scipy.spatial.distance import cdist

    pred   = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 or target.sum() == 0:
        return float('nan')

    def surface_points(mask):
        interior = binary_erosion(mask)
        surface  = mask & ~interior
        pts = np.argwhere(surface).astype(float)
        pts *= np.array(spacing)
        return pts

    s1 = surface_points(pred)
    s2 = surface_points(target)

    dists = cdist(s1, s2)

    hd1 = np.percentile(dists.min(axis=1), 95)
    hd2 = np.percentile(dists.min(axis=0), 95)

    return float(max(hd1, hd2))
