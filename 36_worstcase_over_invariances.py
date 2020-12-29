"""
Basically just the inner-loop of Algorithm 3.
"""

import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import lib
from lib import (ops, utils, datasets, pca, adversarial_translation,
    hparam_search)
import argparse
import geomloss
import time
import sys
import copy
from typing import List
import collections

BATCH_SIZE = 1024
DISC_DIM = 512
N_INVARIANCES = 64
N_TOP = 8
PCA_DIM = 128
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
parser.add_argument('--hparam_search', action='store_true')
parser.add_argument('--z_dim', type=int, default=8)
args = parser.parse_args()
print('Args:')
for k,v in sorted(vars(args).items()):
    print(f'\t{k}: {v}')

def loss_wrt_invariance(invariance_logits, classifier_logits):
    """
    invariance_logits: (n_instances, batch_size, n_classes)
    classifier_logits: (n_instances, batch_size, n_classes)
    result: (n_instances,)
    """
    invariance_probs = F.softmax(invariance_logits, dim=-1)
    result = ops.softmax_cross_entropy(classifier_logits, invariance_probs)
    return result.mean(dim=1)

def trial_fn(**hparams):
    print('Hparams:')
    for k,v in sorted(hparams.items()):
        print(f'\t{k}: {v}')

    X_source, y_source, X_target, y_target = datasets.colored_mnist()

    source_pca = pca.PCA(X_source, PCA_DIM, whiten=True)
    target_pca = pca.PCA(X_target, PCA_DIM, whiten=True)
    X_source = source_pca.forward(X_source)
    X_target = target_pca.forward(X_target)

    # Apply random orthogonal transforms for optimization reasons.
    W1 = ops.random_orthogonal_matrix(PCA_DIM)
    W2 = ops.random_orthogonal_matrix(PCA_DIM)
    X_source = X_source @ W1.T
    X_target = X_target @ W2.T


    source_rep, target_rep, dann_classifier, divergences, accs = (
        adversarial_translation.dann(
            X_source, y_source, X_target, y_target, N_INVARIANCES,
            batch_size=BATCH_SIZE,
            disc_dim=DISC_DIM,
            steps=STEPS,
            z_dim=args.z_dim,
            **hparams))

    best = torch.argsort(divergences)[:N_TOP]
    Xt = X_target.expand(N_INVARIANCES, -1, -1)
    invariance_logits = dann_classifier(target_rep(Xt))
    invariance_logits = invariance_logits[best,:,:].clone().detach()

    matrix = torch.zeros((N_TOP, N_TOP)).cuda()
    for i in range(N_TOP):
        for j in range(N_TOP):
            matrix[i, j] = loss_wrt_invariance(
                invariance_logits[i][None,:],
                invariance_logits[j][None,:]
            )[0].detach()
    # (i,j)th entry is the cross-entropy of classifier j wrt the predictive
    # distribution of classifier i
    print('Invariance classifier cross-entropies:')
    print(matrix)

    classifier = nn.Linear(PCA_DIM, int(y_source.max()+1), bias=False).cuda()

    def forward():
        classifier_logits = classifier(X_target[None,:,:])
        return loss_wrt_invariance(invariance_logits, classifier_logits).max()
    utils.train_loop(forward, optim.Adam(classifier.parameters()), 10001)

    test_acc = ops.multiclass_accuracy(classifier(X_target), y_target).mean()
    print(f'Test acc: {test_acc}')

    return test_acc

if args.hparam_search:
    hparam_search.hparam_search(trial_fn, DEFAULT_HPARAMS)
else:
    trial_fn(**DEFAULT_HPARAMS)