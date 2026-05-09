"""Util functions
Extended from original PANet code
TODO: move part of dataset configurations to data_utils
"""
import random
import torch
import numpy as np
import operator
from scipy.ndimage import distance_transform_edt

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 0
    return fg_bbox, bg_bbox

def t2n(img_t):
    """
    torch to numpy regardless of whether tensor is on gpu or memory
    """
    if img_t.is_cuda:
        return img_t.data.cpu().numpy()
    else:
        return img_t.data.numpy()

def to01(x_np):
    """
    normalize a numpy to 0-1 for visualize
    """
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-5)

def compose_wt_simple(is_wce, data_name):
    """
    Weights for cross-entropy loss
    """
    if is_wce:
        if data_name in ['SABS', 'SABS_Superpix', 'C0', 'C0_Superpix', 'CHAOST2', 'CHAOST2_Superpix', 'CARDIAC_bssFP', 'CARDIAC_LGE']:
            return torch.FloatTensor([0.05, 1.0]).cuda()
        else:
            raise NotImplementedError
    else:
        return torch.FloatTensor([1.0, 1.0]).cuda()


class CircularList(list):
    """
    Helper for spliting training and validation scans
    Originally: https://stackoverflow.com/questions/8951020/pythonic-circular-list/8951224
    """
    def __getitem__(self, x):
        if isinstance(x, slice):
            return [self[x] for x in self._rangeify(x)]

        index = operator.index(x)
        try:
            return super().__getitem__(index % len(self))
        except ZeroDivisionError:
            raise IndexError('list index out of range')

    def _rangeify(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if step is None:
            step = 1
        return range(start, stop, step)


def compute_sdf_from_mask(masks, K_bins=10):
    """
    masks: (K, H, W) float tensor, values 0/1
    returns: (K, 1, H, W) float tensor, values 0~K_bins-1
    """
    K, H, W = masks.shape
    sdf_maps = torch.zeros(K, H, W, dtype=torch.float32)
    for k in range(K):
        mask_np = masks[k].cpu().numpy().astype(np.uint8)
        if mask_np.sum() == 0:
            continue
        dist = distance_transform_edt(mask_np).astype(np.float32)
        max_dist = dist.max()
        if max_dist > 0:
            scale = np.floor(dist / max_dist * (K_bins - 1)).astype(np.int32)
            scale = np.clip(scale, 0, K_bins - 1)
        else:
            scale = np.zeros_like(dist, dtype=np.int32)
        scale = scale * mask_np
        sdf_maps[k] = torch.from_numpy(scale.astype(np.float32))
    return sdf_maps.unsqueeze(1)  # (K, 1, H, W)

