"""
Experiment configuration file
Extended from config file from original PANet Repository
"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds


sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('mySSL')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    """Default configurations"""
    # ADNet setting
    gpu_id         = 0
    n_sv           = 5000
    min_size       = 200
    max_slices     = 3
    use_gt         = True
    exclude_label  = None
    train_organ    = None
    supp_idx       = 0

    # evaluation
    eval_domains   = ['SABS', 'CHAOST2', 'MI-PRO', 'CARDIAC_bssFP', 'CARDIAC_LGE', 'ISIC', 'Lung']

    use_coco_init = True
    usealign = True


    # Network
    modelname = 'dlfcn_res101' # 'dlfcn_res50'
    reload_model_path = '/change/this/to/your/own/path'
    proto_grid_size = 8
    feature_hw = [32, 32]


    model = {
        'align': usealign,
        'use_coco_init': use_coco_init,
        'which_model': modelname,
        'proto_grid_size': proto_grid_size,
        'feature_hw': feature_hw,
        'reload_model_path': reload_model_path,
        'shape_branch': {
            'enabled': True,
            'feat_dim': 256,
            'num_organs': 1,
            'K_scale_classes': 10,
            'geo_emb_hidden': 64,
        },
    }

    path = {
        'log_dir': './runs',
        'SABS': {
            'data_dir':      './Geoproto/data/ABD/ABDOMEN_CT/sabs_CT_normalized',
            'test_label':    [6, 2, 3, 1],
        },
        'CHAOST2': {
            'data_dir':      './Geoproto/data/ABD/ABDOMEN_MR/chaos_MR_T2_normalized',
            'test_label':    [1, 2, 3, 4],
        },
        'CARDIAC_bssFP': {
            'data_dir': './Geoproto/data/Cardiac/bSSFP/cmr_bssFP_normalized',
            'test_label':    [1, 2, 3],
        },
        'CARDIAC_LGE': {
            'data_dir': './Geoproto/data/Cardiac/LGE/cmr_LGE_normalized',
            'test_label':     [1, 2, 3],
        },
        'MI-PRO': {
            'data_dir': './Geoproto/data/MI-PRO/normalized',
            'test_label':     [1, 5, 6],
        },
        'ISIC': {
            'data_dir': './Geoproto/data',
        },
        'Lung': {
            'data_dir': './Geoproto/data',
        },
        # ─────────────────────────────────────────────────────────────────────
    }
    # please put your own data path here
