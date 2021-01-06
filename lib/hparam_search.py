import collections
import copy
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

def _apply_hparam_noise(hparams, noise_std):
    hparams = copy.deepcopy(hparams)
    for name in hparams.keys():
        hparam_type = type(hparams[name])
        hparam = np.log10(hparams[name])
        hparam += np.random.randn() * noise_std
        hparam = 10**hparam
        hparams[name] = hparam_type(hparam)
    return hparams

def _random_search(description, trial_fn, default_hparams, std, n_trials):
    best_score = -np.inf
    best_hparams = None
    for trial in range(n_trials):
        trial_std = (0. if trial==0 else std)
        trial_hparams = _apply_hparam_noise(default_hparams, trial_std)

        print(f'{description}, trial {trial+1} of {n_trials}. Hparams:')
        for k,v in sorted(trial_hparams.items()):
            print(f'\t{k}: {v}')

        score = trial_fn(**trial_hparams).item()
        if score > best_score:
            best_score = score
            best_hparams = trial_hparams

        print('Score:', score)
        print('-'*80)
    return best_hparams

def hparam_search(trial_fn, default_hparams):
    best_hparams = _random_search('Phase 1', trial_fn, default_hparams, 0.5, 15)
    best_hparams = _random_search('Phase 2', trial_fn, best_hparams, 0.2, 15)
    return best_hparams