import os
import warnings
import argparse

from src.train import execute

if __name__ == '__main__':
    resume_id = None
    resume_check = None
    configs = {
        'baseline':                     False,  # 1x (old)
        'baseline_decay':               False,  # 2x
        'pruning':                      False,  # 1x (A)
        'pruning_decay':                False,  # 2x (B)
        'pruning_predecay':             False,  # 1x (2. running)
        'neglecting':                   False,  # 1x
        'quantize':                     False,  # 2x
        'spike_pruning':                True,  # 0x (1. running)
        'spike_distill':                False,   # 0x (1. running)

        'distill':                      False,
        'spike':                        False,
        'pruning_quantize':             False,
        'pruning_spiking':              False,
        'pruning_distill':              False,
        'quantize_distill':             False,
        'pruning_quantize_distill':     False,
        'pruning_spiking_distill':      False,
    }

    parser = argparse.ArgumentParser(description='Run configurations')
    parser.add_argument('-c', '--configs', help='list of configurations to run', nargs='+', default=[])
    parser.add_argument('-p', '--path', help='Path to run configurations', default='./config')
    parser.add_argument('-r', '--repeats', help='Number of repeats per configuration', default=1)
    parser.add_argument('--multicache', action='store_true', help="Use parallel processing when caching dataset")
    parser.add_argument('-i', '--id', help='Resume ID for logging', default=resume_id)
    parser.add_argument('-s', '--checkpoint', help='Resume checkpoint', default=resume_check)
    parser.add_argument('-e', '--epoch', help='Initial epoch to start from', default=1)
    args = parser.parse_args()

    if len(args.configs) == 0:
        runs = [cfg for cfg, act in configs.items() if act]
    else:
        runs = args.configs
    print('Run configurations:', ', '.join(runs))
    print()

    for config_name in runs:
        cfg = os.path.join(args.path, config_name + '.json')
        if os.path.isfile(cfg):
            for run_idx in range(args.repeats):
                execute(cfg, root='./', run_name_index=run_idx+1, multicache=args.multicache,
                        resume=args.id, checkpoint=args.checkpoint, init_epoch=args.epoch)
        else:
            warnings.warn(f'Path {cfg} does not exist. Skipping...')