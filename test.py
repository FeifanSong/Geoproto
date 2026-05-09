"""
Multi-domain validation using ADNet TestDataset protocol.
Usage:
  python test.py
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.shapeproto import FewShotSeg
from dataloaders.dataset_specifics import *
from dataloaders.isic import DatasetISIC
from dataloaders.lung import DatasetLung
from config_for_test import ex
import glob, SimpleITK as sitk
import torch.nn.functional as F

os.environ['TORCH_HOME'] = './pretrained_model'

# Maps Sacred config dataset names to ADNet internal names.
# 3-D nii domain --> eval_domain
_DATASET_NAME_MAP = {
    'CHAOST2':       'ABDOMEN_MR',
    'SABS':          'ABDOMEN_CT',
    'CARDIAC_bssFP': 'CARDIAC_bssFP',
    'CARDIAC_LGE':   'CARDIAC_LGE',
    'MI-PRO':        'MI-PRO',
}
# 2-D PNG domain --> eval_domain_2d
_2D_DOMAINS = {'ISIC', 'Lung'}
# Note! target size should be aligned with source size
_TARGET_SIZE = 257
# ─────────────────────────────────────────────────────────────────────────────

def _adnet_dataset_and_dir(config_dataset, config_data_dir):
    adnet_name = _DATASET_NAME_MAP.get(config_dataset, config_dataset)
    data_dir   = os.path.normpath(config_data_dir)
    if adnet_name in ('CARDIAC_bssFP', 'CARDIAC_LGE', 'MI-PRO', 'ABDOMEN_MR', 'ABDOMEN_CT'):
        img_pattern = os.path.join(data_dir, 'image*.nii.gz')
    else:
        img_pattern = os.path.join(data_dir, '*/image*.nii.gz')
    return adnet_name, img_pattern


def resize_vol(vol, target=_TARGET_SIZE):
    if vol.shape[1] == target and vol.shape[2] == target:
        return vol
    t = torch.from_numpy(vol.astype(np.float32)).unsqueeze(1)
    t = torch.nn.functional.interpolate(
        t, size=(target, target), mode='bilinear', align_corners=False)
    return t[:, 0].numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  3-D NIfTI evaluation — CHAOST2 / SABS / CARDIAC / MI-PRO
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_domain(model, data_dir, device, domain_name, test_label):

    adnet_name, adnet_dir = _adnet_dataset_and_dir(domain_name, data_dir)

    print(f'  img_pattern inside eval_domain: {adnet_dir}')
    print(f'  glob result: {glob.glob(adnet_dir)[:3]}')

    image_files = sorted(glob.glob(adnet_dir),
                         key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
    print(f'  [{domain_name}] found {len(image_files)} scans')

    # Get unique labels (classes).
    labels = get_label_names(adnet_name)
    class_dice = {name: [] for name in labels.values()}

    for img_path in image_files:
        lb_path = img_path.replace('image_', 'label_')
        if not os.path.exists(lb_path):
            continue
        scan_id = os.path.basename(img_path).replace('image_', '').replace('.nii.gz', '')
        img_vol = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
        lb_vol  = sitk.GetArrayFromImage(sitk.ReadImage(lb_path)).astype(np.int32)

        img_vol = resize_vol(img_vol)
        lb_vol  = resize_vol(lb_vol)
        img_vol = (img_vol - img_vol.mean()) / (img_vol.std() + 1e-8)
        # print(img_vol.shape)

        for label_val, label_name in labels.items():
            # Skip BG class. Only test fot test_label
            if label_name == 'BG':
                continue
            elif np.intersect1d([label_val], test_label).size == 0:
                continue

            lb_bin    = (lb_vol == label_val).astype(np.int32)
            fg_slices = np.where(lb_bin.sum(axis=(1, 2)) > 0)[0]
            if len(fg_slices) < 2:
                continue

            mid         = fg_slices[len(fg_slices) // 2]
            sup_img_np  = img_vol[mid]
            sup_mask_np = lb_bin[mid].astype(np.float32)

            sup_img_t   = torch.from_numpy(sup_img_np).unsqueeze(0).unsqueeze(0).float().to(device)
            sup_img_3ch = sup_img_t.repeat(1, 3, 1, 1).unsqueeze(0)
            sup_mask_t  = torch.from_numpy(sup_mask_np).unsqueeze(0).unsqueeze(0).to(device)

            support_images  = [[sup_img_3ch[0]]]
            support_fg_mask = [[sup_mask_t[0]]]
            support_bg_mask = [[(1.0 - sup_mask_t[0])]]

            query_slices = [s for s in fg_slices if s != mid]  # remove support slice!
            slice_dices  = []

            for q_z in query_slices:
                q_img_np  = img_vol[q_z]
                q_mask_np = lb_bin[q_z].astype(np.float32)
                q_img_t   = torch.from_numpy(q_img_np).unsqueeze(0).unsqueeze(0).float().to(device)
                q_img_3ch = q_img_t.repeat(1, 3, 1, 1)

                query_pred, _, _ = model(
                    support_images, support_fg_mask, support_bg_mask,
                    [q_img_3ch], isval=True, val_wsize=2,
                )

                pred_mask = query_pred.argmax(dim=1)[0].cpu().numpy()

                pred_bool = pred_mask.astype(bool)
                gt_bool   = q_mask_np.astype(bool)
                inter     = (pred_bool & gt_bool).sum()
                denom     = pred_bool.sum() + gt_bool.sum()
                dsc       = (2.0 * inter / denom) if denom > 0 else 1.0
                slice_dices.append(float(dsc))

            if slice_dices:
                mean_dsc = float(np.mean(slice_dices))
                class_dice[label_name].append(mean_dsc)
                print(f'    scan {scan_id} {label_name}: DSC={mean_dsc:.4f} ({len(slice_dices)} slices)')

    result  = {}
    all_dsc = []
    for label_name, dices in class_dice.items():
        if dices:
            m = float(np.mean(dices))
            result[label_name] = m
            all_dsc.extend(dices)
            print(f'  {label_name}: mean DSC = {m:.4f} ({len(dices)} scans)')
        else:
            result[label_name] = float('nan')

    overall = float(np.nanmean(all_dsc)) if all_dsc else float('nan')

    return result, overall


# ══════════════════════════════════════════════════════════════════════════════
#  2-D PNG evaluation — ISIC / Lung
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_domain_2d(model, domain_name, data_dir, device):
    """
    2-D few-shot evaluation for PNG datasets (ISIC / Lung).
    """
    # ── Setup Dataset ──────────────────────────────────────────────────────────
    if domain_name == 'ISIC':
        dataset = DatasetISIC(datapath = data_dir, split = 'val')
        category_names = {0: 'nevus', 1: 'melanoma', 2: 'seborrheic_keratosis'}

    elif domain_name == 'Lung':
        dataset = DatasetLung(datapath = data_dir, split = 'val')
        category_names = {0: 'lung'}

    else:
        raise ValueError(f'Unknown 2-D domain: {domain_name}')

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True)

    # ── Class Dice ───────────────────────────────────────────────────────
    class_dice = {name: [] for name in category_names.values()}

    for batch in loader:
        sup_imgs  = batch['support_imgs'].to(device)
        sup_masks = batch['support_masks'].to(device)
        qry_img   = batch['query_img'].to(device)
        qry_mask  = batch['query_mask'].to(device)
        class_id  = batch['class_id'].item()

        if sup_imgs.dim() == 4:
            sup_imgs  = sup_imgs.unsqueeze(1)   # (1, 1, 3, H, W)
            sup_masks = sup_masks.unsqueeze(1)  # (1, 1, H, W)

        n_shot_actual = sup_imgs.shape[1]

        # model.forward()  list-of-lists
        support_images  = [[sup_imgs[s]          for s in range(n_shot_actual)]]
        support_fg_mask = [[sup_masks[s]         for s in range(n_shot_actual)]]
        support_bg_mask = [[(1.0 - sup_masks[s]) for s in range(n_shot_actual)]]

        query_pred, _, _ = model(
            support_images, support_fg_mask, support_bg_mask,
            [qry_img], isval=True, val_wsize=2,
        )

        pred_mask = query_pred.argmax(dim=1)[0].cpu().numpy().astype(bool)
        gt_mask   = qry_mask[0].cpu().numpy().astype(bool)

        inter = (pred_mask & gt_mask).sum()
        denom = pred_mask.sum() + gt_mask.sum()
        dsc   = float(2.0 * inter / denom) if denom > 0 else 1.0

        cat_name = category_names.get(class_id, str(class_id))
        class_dice[cat_name].append(dsc)

    # ── Conclusion ──────────────────────────────────────────────────────────────────
    result  = {}
    all_dsc = []
    for cat_name, dices in class_dice.items():
        if dices:
            m = float(np.mean(dices))
            result[cat_name] = m
            all_dsc.extend(dices)
            print(f'  {cat_name}: mean DSC = {m:.4f} ({len(dices)} episodes)')
        else:
            result[cat_name] = float('nan')

    overall = float(np.nanmean(all_dsc)) if all_dsc else float('nan')
    return result, overall


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

@ex.automain
def main(_run, _config, _log):
    device = torch.device(
        f"cuda:{_config['gpu_id']}" if torch.cuda.is_available() else 'cpu'
    )

    model = FewShotSeg(
        pretrained_path=_config['reload_model_path'],
        cfg=_config['model'],
    ).to(device)
    model.eval()

    summary = {}
    for domain_name in _config['eval_domains']:
        data_dir = _config['path'][domain_name]['data_dir']
        print(f'Evaluating domain {domain_name}, data dir {data_dir}')
        if not data_dir or not os.path.exists(data_dir):
            print(f'[warn] {domain_name} data_dir not found, skipping')
            continue

        _log.info(f'###### Evaluating {domain_name} ######')

        if domain_name in _2D_DOMAINS:
            class_dice, mean_dsc = eval_domain_2d(
                model       = model,
                domain_name = domain_name,
                data_dir    = data_dir,
                device      = device,
            )
        else:
            class_dice, mean_dsc = eval_domain(
                model        = model,
                data_dir     = data_dir,
                device       = device,
                domain_name  = domain_name,
                test_label   = _config['path'][domain_name]['test_label'],
            )

        summary[domain_name] = mean_dsc
        for cls, dsc in class_dice.items():
            _run.log_scalar(f'{domain_name}/{cls}_dsc', dsc)
        _run.log_scalar(f'{domain_name}/mean_dsc', mean_dsc)

    print('\n========== Summary ==========')
    for domain, dsc in summary.items():
        print(f'{domain:20s}  mean DSC = {dsc:.4f}')

    _log.info('End of multi-domain validation')

    return summary