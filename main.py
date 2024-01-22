import os
import warnings
import argparse

from src.train import execute

if __name__ == '__main__':
    configs = {
        'baseline':                     True,
        'baseline_decay':               True,
        'pruning':                      False,
        'pruning_decay':                False,
        'quantize':                     False,
        'distill':                      False,
        'spiking':                      False,
        'neglecting':                   False,
        'pruning_quantize':             False,
        'pruning_spiking':              False,
        'pruning_distill':              False,
        'quantize_distill':             False,
        'spiking_distill':              False,
        'pruning_quantize_distill':     False,
        'pruning_spiking_distill':      False,
    }

    parser = argparse.ArgumentParser(description='Run configurations')
    parser.add_argument('-c', '--configs', help='list of configurations to run', nargs='+', default=[])
    parser.add_argument('-p', '--path', help='Path to run configurations', default='./config')
    parser.add_argument('-r', '--repeats', help='Number of repeats per configuration', default=1)
    parser.add_argument('--multicache', action='store_true', help="Use parallel processing when caching dataset")
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
                execute(cfg, root='./', run_name_index=run_idx+1, multicache=args.multicache)
        else:
            warnings.warn(f'Path {cfg} does not exist. Skipping...')