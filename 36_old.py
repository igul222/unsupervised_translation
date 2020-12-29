"""
Basically just the inner-loop of Algorithm 3.
"""

import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import lib
from lib import ops, utils, datasets, pca, worstcase_translation_v2
import argparse
import geomloss
import time
import sys
import copy
from typing import List
import collections

PCA_DIM = 128
LATENT_DIM = 32
N_INSTANCES = 50
N_ROUNDS = 1
BATCH_SIZE = 1024
N_BEST_INVARIANCES = 5
TRANSLATION_STEPS = 10001

def _get_batch(vars_, batch_size, n_instances):
    idx = torch.randint(
        low=0,
        high=len(vars_[0]),
        size=[n_instances * batch_size],
        device='cuda'
    )
    return [v[idx].view(n_instances, batch_size, -1) for v in vars_]

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

classifier = nn.Linear(PCA_DIM, 10, bias=False).cuda()

invariance_logits = []

for round_idx in range(N_ROUNDS):
    print(f'-------- Round {round_idx}:')

    # Find the worst-case-translations
    translation = worstcase_translation_v2.WorstCaseTranslation(
        n_instances=N_INSTANCES, input_dim=PCA_DIM, latent_dim=LATENT_DIM,
        steps=TRANSLATION_STEPS
    )
    energy_dists, worstcase_losses = translation.train(
        X_source, y_source, X_target, classifier, verbose=True
    )

    utils.print_row('instance', 'energy_dist', 'worstcase_loss')
    for idx in torch.argsort(energy_dists):
        utils.print_row(idx, energy_dists[idx], worstcase_losses[idx])

    for idx in torch.argsort(energy_dists)[:N_BEST_INVARIANCES]:
        Zt = X_target @ translation.target_rep.weight[idx].T
        invariance_logits_ = Zt @ translation.classifier.weight[idx].T
        invariance_logits_ += translation.classifier.bias[idx,None,:]
        invariance_logits.append(invariance_logits_.clone().detach())

    matrix = torch.zeros((len(invariance_logits), len(invariance_logits)), device='cuda')
    for inv_idx in range(len(invariance_logits)):
        for classifier_idx in range(len(invariance_logits)):
            matrix[inv_idx, classifier_idx] = worstcase_translation_v2.calculate_worstcase_loss_2(
                invariance_logits[inv_idx][None,:],
                invariance_logits[classifier_idx][None,:]
            )[0].detach()
    # (i,j)th entry is the cross-entropy of classifier j wrt the predictive
    # distribution of classifier i
    print('Invariance classifier cross-entropies:')
    print(matrix)

    def forward():
        idx = torch.randint(low=0, high=len(X_target), size=[BATCH_SIZE],
            device='cuda')
        inv_logits = torch.stack(
            [logits[idx] for logits in invariance_logits],
            dim=0)
        classifier_logits = classifier(X_target[None, idx])

        return worstcase_translation_v2.calculate_worstcase_loss_2(
            inv_logits,
            classifier_logits
        ).max()

    # utils.train_loop(classifier, forward, 10001, 1e-3)
    utils.train_loop(forward, optim.Adam(classifier.parameters(), lr=1e-3), 10001)

    test_acc = ops.multiclass_accuracy(classifier(X_target), y_target).mean()
    print(f'Test acc: {test_acc}')
