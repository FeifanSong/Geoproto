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

ex = Experiment('Geoproto')
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
    n_sv           = 5000
    min_size       = 200
    use_gt         = False  # use superpixel pseudo mask
    test_label     = [1, 2, 3, 6]
    exclude_label  = None

    seed = 1234
    gpu_id = 0
    num_workers = 4 # 0 for debugging. 

    dataset = 'SABS' # i.e. abdominal ct
    use_coco_init = True # initialize backbone with MS_COCO initialization. Anyway coco does not contain medical images

    ### Training
    n_steps = 50100
    batch_size = 1
    lr_milestones = [ (ii + 1) * 1000 for ii in range(n_steps // 1000 - 1)]
    lr_step_gamma = 0.95
    ignore_label = 255
    print_interval = 100
    save_snapshot_every = 5000
    max_iters_per_load = 1000 # epoch size, interval for reloading the dataset
    usealign = True # see vanilla PANet
    use_wce = True


    # Network
    modelname = 'dlfcn_res101' # resnet 101 backbone from torchvision fcn-deeplab
    proto_grid_size = 8 # L_H, L_W = (32, 32) / 8 = (4, 4)  in training
    feature_hw = [32, 32] # feature map size, should couple this with backbone in future

    # SSL
    superpix_scale = 'MIDDLE' #MIDDLE/ LARGE
    lambda_sdf      = 0.3    # weight for SDFLoss (DDT)
    lambda_sdf_dist = 1.0    # weight for distance penalty term inside SDFLoss 1.0
    lambda_shape    = 1.0    # weight for shape-branch CE loss Default: 1.0


    # Default
    model = {
        'align': usealign,
        'use_coco_init': use_coco_init,
        'which_model': modelname,
        'proto_grid_size': proto_grid_size,
        'feature_hw': feature_hw,
        'shape_branch': {
            'enabled': True,
            'feat_dim': 256,
            'num_organs': 1,
            'K_scale_classes': 10,
            'geo_emb_hidden': 64,
        },
    }

    task = {
        'n_ways': 1,
        'n_shots': 1,
        'n_queries': 1,
    }

    optim = {
        'lr': 1e-3, 
        'momentum': 0.9,
        'weight_decay': 0.0005,
    }

    exp_prefix = ''

    exp_str = '_'.join(
        [exp_prefix]
        + [dataset,]
        + [f'{task["n_shots"]}shot'])

    path = {
        'log_dir': './runs',
        # 'SABS':{'data_dir': "./Geoproto/data/ABD/ABDOMEN_CT/sabs_CT_normalized"},
        'SABS': {'data_dir': "/media/cs4007/disk2/ShapeprotoV4/Self-supervised-Fewshot-Medical-Image-Segmentation-master/data/ABD"
                             "/ABDOMEN_CT/sabs_CT_normalized"},
        'CHAOST2':{'data_dir': "./Geoproto/data/ABD/ABDOMEN_MR/chaos_MR_T2_normalized"},
        'CARDIAC_bssFP': {'data_dir': './Geoproto/data/Cardiac/bSSFP/cmr_bssFP_normalized'},
        'CARDIAC_LGE': {'data_dir': './Geoproto/data/Cardiac/LGE/cmr_LGE_normalized'},
        }
    # please put your own data path here


@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
