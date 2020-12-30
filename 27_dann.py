"""
Evaluation of DANN with orthogonal featurizers. We run a random hparam search
centered around hand-tuned defaults, and then another fine-grained search
centered around the best-performing hparams from the first search.
"""
import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
from lib import (ops, utils, datasets, pca, hparam_search,
    adversarial_translation)
import argparse
import collections
import copy

PCA_DIM = 128
DISC_DIM = 512
BATCH_SIZE = 1024
STEPS = 10001
DEFAULT_HPARAMS = {
    'lr_g': 1e-3,
    'lr_d': 5e-4,
    'lambda_erm': 5.0,
    'lambda_gp': 10.0,
    'lambda_orth': 0.01,
    'weight_decay_d': 1e-5
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='colored_mnist')
parser.add_argument('--hparam_search', action='store_true')
parser.add_argument('--n_instances', type=int, default=64)
parser.add_argument('--n_top', type=int, default=8)
parser.add_argument('--z_dim', type=int, default=32)
args = parser.parse_args()

print('Args:')
for k,v in sorted(vars(args).items()):
    print(f'\t{k}: {v}')

def trial_fn(**hparams):
    print('Hparams:')
    for k,v in sorted(hparams.items()):
        print(f'\t{k}: {v}')

    X_source, y_source, X_target, y_target = datasets.REGISTRY[args.dataset]()
    source_pca = pca.PCA(X_source, PCA_DIM, whiten=True)
    target_pca = pca.PCA(X_target, PCA_DIM, whiten=True)
    X_source = source_pca.forward(X_source)
    X_target = target_pca.forward(X_target)
    # Apply random orthogonal transforms for optimization reasons.
    W1 = ops.random_orthogonal_matrix(X_source.shape[1])
    W2 = ops.random_orthogonal_matrix(X_target.shape[1])
    X_source = X_source @ W1.T
    X_target = X_target @ W2.T

    source_rep, target_rep, classifier, divergences, accs = (
        adversarial_translation.dann(
            X_source, y_source, X_target, y_target, args.n_instances,
            batch_size=BATCH_SIZE,
            disc_dim=DISC_DIM,
            steps=STEPS,
            z_dim=args.z_dim,
            **hparams))

    print(f'All accs: {accs.mean().item()} +/- {accs.std().item()}')
    top_accs = accs[torch.argsort(divergences)][:args.n_top]
    top_accs_mean = top_accs.mean().item()
    top_accs_std = top_accs.std().item()
    print(f'Top {args.n_top} accs: {top_accs_mean} +/- {top_accs_std}')
    return top_accs_mean

if args.hparam_search:
    hparam_search.hparam_search(trial_fn, DEFAULT_HPARAMS)
else:
    trial_fn(**DEFAULT_HPARAMS)