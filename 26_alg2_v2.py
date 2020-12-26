"""
Attempt 2 at implementing Algorithm 2
"""

import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import lib
from lib import ops, utils, datasets, pca, worstcase_translation
import argparse
import geomloss
import time
import sys
import copy
from typing import List
import collections

LATENT_DIM = 32
PCA_DIM = 64
N_INSTANCES = 8
LR_G = 1e-3
LR_D = 1e-3
D_STEPS = 10
DISC_DIM = 512

X_source, y_source, X_target, y_target = datasets.colored_mnist()
X_target, y_target, X_test, y_test = datasets.split(X_target, y_target, 0.9)

source_pca = pca.PCA(X_source, PCA_DIM, whiten=True)
target_pca = pca.PCA(X_target, PCA_DIM, whiten=True)
X_source = source_pca.forward(X_source)
X_target = target_pca.forward(X_target)
X_test = target_pca.forward(X_test)

disc = nn.Sequential(
    nn.Linear(LATENT_DIM, DISC_DIM),
    nn.GELU(),
    nn.Linear(DISC_DIM, DISC_DIM),
    nn.GELU(),
    nn.Linear(DISC_DIM, 1)
).cuda()

# Apply random orthogonal transforms for optimization reasons.
W1 = ops.random_orthogonal_matrix(PCA_DIM)
W2 = ops.random_orthogonal_matrix(PCA_DIM)
X_source = X_source @ W1.T
X_target = X_target @ W2.T
X_test   = X_test   @ W2.T

source_rep = nn.Linear(PCA_DIM, LATENT_DIM, bias=False).cuda()
torch.nn.init.orthogonal_(source_rep.weight)
classifier = nn.Linear(LATENT_DIM, 10, bias=False).cuda()
target_rep = nn.Linear(PCA_DIM, LATENT_DIM, bias=False).cuda()
torch.nn.init.orthogonal_(target_rep.weight)

gen_opt = optim.Adam(
    list(source_rep.parameters()) +
    list(target_rep.parameters()) +
    list(classifier.parameters()),
    lr=LR_G, betas=(0.5, 0.99)
)

disc_opt = optim.Adam(disc.parameters(), lr=LR_D, betas=(0.5, 0.99))

worstcase = worstcase_translation.WorstCaseTranslation(
    n_instances=8, input_dim=PCA_DIM, latent_dim=LATENT_DIM,
    lambda_tv=0.0)

def disc_forward():
    Z_source = source_rep(X_source)
    Z_target = target_rep(X_target)

    disc_source = disc(Z_source)
    disc_target = disc(Z_target)

    disc_loss = F.binary_cross_entropy_with_logits(
        disc_source, torch.ones_like(disc_source))
    disc_loss = disc_loss + F.binary_cross_entropy_with_logits(
        disc_target, torch.zeros_like(disc_target))
    disc_loss = disc_loss / 2.

    grad_s = autograd.grad(disc_source.sum(), [Z_source], create_graph=True)[0]
    grad_t = autograd.grad(disc_target.sum(), [Z_target], create_graph=True)[0]
    grad_penalty = grad_s.square().sum(dim=1).mean()
    grad_penalty = grad_penalty + grad_t.square().sum(dim=1).mean()

    return disc_loss + grad_penalty

def classifier_forward():
    return F.cross_entropy(classifier(source_rep(X_source)), y_source)

def worstcase_forward():
    # return torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
    alt_target_reps, energy_dists, tvs = worstcase.train(
        X_source, X_target, source_rep, target_rep, classifier, verbose=False)

    best_idx = torch.argsort(energy_dists)[0]

    best_tvs = []
    best_energy_dists = []
    for idx in torch.argsort(energy_dists)[:1]:
        Z_target = target_rep(X_target)
        Z_target_alt = X_target @ alt_target_reps.weight[idx].T
        preds = F.softmax(classifier(Z_target), dim=1)
        alt_preds = F.softmax(classifier(Z_target_alt), dim=1)
        tv = (preds - alt_preds).abs().max(dim=1)[0].mean()
        energy_dist = energy_dists[idx]
        best_tvs.append(tv)
        best_energy_dists.append(energy_dist)

    return torch.stack(best_tvs).mean(), torch.stack(best_energy_dists).mean()

def calculate_energy_dist(X_source, X_target):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            X = source_rep(X_source)
            Y = target_rep(X_target)
            DXX = torch.cdist(X, X).mean()
            DYY = torch.cdist(Y, Y).mean()
            DXY = torch.cdist(X, Y).mean()
            return (DXY - DXX + DXY - DYY)

utils.print_row('step', 'disc_loss', 'classifier_loss', 'tv_loss',
    'p2_energy_dist', 'p1_energy_dist', 'test acc')
histories = collections.defaultdict(lambda: [])
for step in range(1000):
    utils.enforce_orthogonality(source_rep.weight)
    utils.enforce_orthogonality(target_rep.weight)

    for _ in range(D_STEPS):
        disc_opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            disc_loss = disc_forward()
        disc_loss.backward()
        disc_opt.step()

    disc_opt.zero_grad(set_to_none=True)
    gen_opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        disc_loss = disc_forward()
        classifier_loss = classifier_forward()
        tv_loss, energy_dist = worstcase_forward()
    ((0.1*-disc_loss) + (0.1*classifier_loss) + (0.1*tv_loss)).backward()
    gen_opt.step()

    histories['disc_loss'].append(disc_loss.item())
    histories['classifier_loss'].append(classifier_loss.item())
    histories['tv_loss'].append(tv_loss.item())
    histories['energy_dist'].append(energy_dist.item())

    if step % 10 == 0:
        test_acc = ops.multiclass_accuracy(
            classifier(target_rep(X_test)), y_test).mean()
        utils.print_row(
            step,
            np.mean(histories['disc_loss']),
            np.mean(histories['classifier_loss']),
            np.mean(histories['tv_loss']),
            np.mean(histories['energy_dist']),
            calculate_energy_dist(X_source, X_target),
            test_acc.item()
        )
        histories.clear()
