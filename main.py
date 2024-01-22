import os
import warnings

from src.train import execute

if __name__ == '__main__':
    config_path = './config'

    configs = {
        'baseline':                     False,
        'pruning':                      True,
        'pruning_decay':                True,
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

    run_repeats = 1

    for (config_name, active) in configs.items():
        if active:
            cfg = os.path.join(config_path, config_name + '.json')
            if os.path.isfile(cfg):
                for run_idx in range(run_repeats):
                    execute(cfg, root='./', run_name_index=run_idx+1)
            else:
                warnings.warn(f'Path {cfg} does not exist. Skipping...')