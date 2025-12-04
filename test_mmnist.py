import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    setattr(args, 'batch_size', 16)
    setattr(args, 'val_batch_size', 16)
    setattr(args, 'dataname', 'mmnist')
    setattr(args, 'method', 'DAMFormer')
    setattr(args, 'model', 'Quadruplet_TSST')
    setattr(args, 'test', True)
    args.model_config = {
        # image h w c
        'height': 64,
        'width': 64,
        'num_channels': 1,
        # video length in and out
        'pre_seq': 10,
        'after_seq': 10,
        # patch size
        'patch_size': 8,
        'dim': 256,
        'heads': 8,
        'dim_head': 32,
        # dropout
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'drop_path': 0.0,
        'scale_dim': 4,
        # depth
        'depth': 1,
        'Ndepth': 6  # TSST
    }

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py')
    loaded_cfg = load_config(cfg_path)
    config = update_config(config, loaded_cfg,
                            exclude_keys=['method', 'batch_size', 'val_batch_size',
                                            'drop_path', 'warmup_epoch'])
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]

    # set multi-process settings
    setup_multi_processes(config)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    
    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()

    if rank == 0 and has_nni:
        nni.report_final_result(mse)