"""
Training the model
Extended from original implementation of PANet by Wang et al.
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np

import torch.nn.functional as F
from models.shapeproto import FewShotSeg
from models.module import SDFLoss
from dataloaders.datasets import TrainDataset
from dataloaders.dataset_specifics import get_label_names
from util.utils import set_seed, t2n, to01, compose_wt_simple, compute_sdf_from_mask

from config_for_train import ex

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"
_train_times = []
_train_mems  = []

# Maps Sacred config dataset names to ADNet internal names.
# ADNet TrainDataset expects the parent dir for ABDOMEN datasets
# (it appends 'chaos_MR_T2_normalized/' or 'sabs_CT_normalized/' internally).
_DATASET_NAME_MAP = {
    'CHAOST2': 'ABDOMEN_MR',
    'SABS':    'ABDOMEN_CT',
    'CARDIAC_bssFP':    'CARDIAC_bssFP',
    'CARDIAC_LGE':      'CARDIAC_LGE',
}

def _adnet_dataset_and_dir(config_dataset, config_data_dir):
    adnet_name = _DATASET_NAME_MAP.get(config_dataset, config_dataset)
    data_dir   = os.path.normpath(config_data_dir)
    if adnet_name in ('ABDOMEN_MR', 'ABDOMEN_CT', 'CARDIAC_bssFP', 'CARDIAC_LGE'):
        data_dir = os.path.dirname(data_dir)
    return adnet_name, data_dir


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    # ── Shape branch Initialization ──────────────────────────────────────────────────────
    _shape_enabled = _config['model'].get('shape_branch', {}).get('enabled', False)
    if _shape_enabled:
        _sdf_mode = _config['model']['shape_branch'].get('sdf_mode', 'fg')
        _out_K = (2 * _config['model']['shape_branch']['K_scale_classes']
                  if _sdf_mode == 'signed'
                  else _config['model']['shape_branch']['K_scale_classes'])
        sdf_criterion = SDFLoss(
            K=_out_K,
            lambda_dist=_config.get('lambda_sdf_dist', 1.0),
        )
        _log.info('###### Shape branch enabled: SDFLoss initialized ######')
    else:
        sdf_criterion = None

    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=None, cfg=_config['model'],
                       sdf_criterion=sdf_criterion)
    model = model.cuda()
    model.train()
    # ──────────────────────────────────────────────────────────────────────────

    _log.info('###### Load data ######')
    _adnet_ds, _adnet_dir = _adnet_dataset_and_dir(
        _config['dataset'], _config['path'][_config['dataset']]['data_dir']
    )
    data_config = {
        'data_dir':      _adnet_dir,
        'dataset':       _adnet_ds,
        'n_shot':        _config['task']['n_shots'],
        'n_way':         _config['task']['n_ways'],
        'n_query':       _config['task']['n_queries'],
        'n_sv':          _config['n_sv'],
        'max_iter':      _config['max_iters_per_load'],
        'min_size':      _config['min_size'],
        'test_label':    _config['test_label'],
        'exclude_label': _config['exclude_label'],
        'use_gt':        _config['use_gt'],
    }
    train_dataset = TrainDataset(data_config)
    trainloader = DataLoader(
        train_dataset,
        batch_size  = _config['batch_size'],
        shuffle     = True,
        num_workers = _config['num_workers'],
        pin_memory  = True,
        drop_last   = True,
    )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=_config['lr_step_gamma'])
    my_weight = compose_wt_simple(_config["use_wce"], _config['dataset'])
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight=my_weight)

    i_iter = 0
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load']
    log_loss = {'loss': 0, 'align_loss': 0, 'sdf_loss': 0}

    _log.info('###### Training ######')
    torch.cuda.reset_peak_memory_stats()
    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### Epoch {sub_epoch}/{n_sub_epoches} ######')
        for _, sample_batched in enumerate(trainloader):
            i_iter += 1

            # ── from TrainDataset batch ──────────────────────────────
            sup_img_list  = sample_batched['support_images'][0]   # (n_way, n_shot, 3, H, W)
            sup_mask_list = sample_batched['support_fg_labels'][0] # (n_way, n_shot, H, W)

            sup_imgs_raw = torch.cat(
                [s[:, 0:1, :, :] for s in sup_img_list], dim=0
            ).unsqueeze(0).float().cuda()              # (1,K,1,H,W)

            sup_masks_raw = torch.cat(
                list(sup_mask_list), dim=0
            ).unsqueeze(0).float().cuda()              # (1,K,H,W)

            q_img_raw = sample_batched['query_images'][0][:, 0:1, :, :].float().cuda()
            # (1,1,H,W)  [n_query=1, gray, H, W]

            query_labels = sample_batched['query_labels'][0].long().cuda()
            if query_labels.dim() == 2:
                query_labels = query_labels.unsqueeze(0)   # (1,H,W)

            # Online compute SDF
            sdf_labels = compute_sdf_from_mask(
                sup_masks_raw[0],   # (K,H,W)
                K_bins=_config['model']['shape_branch']['K_scale_classes'],
            ).unsqueeze(0).cuda()                      # (1,K,1,H,W)

            B, K, _, H, W = sup_imgs_raw.shape

            sup_imgs_3ch = sup_imgs_raw.repeat(1, 1, 3, 1, 1)   # (1,K,3,H,W)
            qry_img_3ch  = q_img_raw.repeat(1, 3, 1, 1)          # (1,3,H,W)

            support_images  = [[sup_imgs_3ch[0, k].unsqueeze(0) for k in range(K)]]
            support_fg_mask = [[sup_masks_raw[0, k].unsqueeze(0) for k in range(K)]]
            support_bg_mask = [[(1.0 - sup_masks_raw[0, k]).unsqueeze(0) for k in range(K)]]
            query_images    = [qry_img_3ch]
            # ────────────────────────────────────────────────────────────

            optimizer.zero_grad()

            try:
                query_pred, align_loss, sdf_loss = model(
                    support_images, support_fg_mask, support_bg_mask,
                    query_images, isval=False, val_wsize=None, sdf_gt=sdf_labels,
                )
            except Exception as e:
                print(f'Faulty batch detected, skip: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()
                continue

            query_loss = criterion(query_pred, query_labels)
            loss = query_loss + align_loss + _config.get('lambda_sdf', 0.3) * sdf_loss

            _run.log_scalar('sdf_loss', float(sdf_loss.item()))
            log_loss['sdf_loss'] += float(sdf_loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()


            # Log loss
            query_loss_val = query_loss.detach().data.cpu().numpy()
            align_loss_val = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

            _run.log_scalar('loss', query_loss_val)
            _run.log_scalar('align_loss', align_loss_val)
            log_loss['loss'] += query_loss_val
            log_loss['align_loss'] += align_loss_val

            if (i_iter + 1) % _config['print_interval'] == 0:
                print(f'step {i_iter+1}: '
                      f'loss={log_loss["loss"] / _config["print_interval"]:.4f}  '
                      f'align={log_loss["align_loss"] / _config["print_interval"]:.4f}  '
                      f'sdf={log_loss["sdf_loss"] / _config["print_interval"]:.4f}  ')
                log_loss = {k: 0 for k in log_loss}

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                if hasattr(trainloader.dataset, 'reload_buffer'):
                    trainloader.dataset.reload_buffer()
                    print(f'###### Dataset reloaded ######')

            if (i_iter - 2) > _config['n_steps']:
                return 1  # finish up

    return 1
