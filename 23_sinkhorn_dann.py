"""
DANN with Sinkhorn divergence.
"""

import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import lib
from lib import ops, utils, datasets, pca
import argparse
import geomloss
import time
import sys

LR = 1e-3
PCA_DIM = 64
LATENT_DIM = 64
BATCH_SIZE = 4096
LAMBDA = 10.0
N_INSTANCES = 16
WHITEN = True
TARGET_ORTHO = True

X_source, y_source, X_target, y_target = datasets.colored_mnist()
X_target, y_target, X_test, y_test = datasets.split(X_target, y_target, 0.9)

source_pca = pca.PCA(X_source, PCA_DIM, whiten=WHITEN)
target_pca = pca.PCA(X_target, PCA_DIM, whiten=WHITEN)
X_source = source_pca.forward(X_source)
X_target = target_pca.forward(X_target)
X_test = target_pca.forward(X_test)

# Apply random orthogonal transforms for optimization reasons.
W1 = ops.random_orthogonal_matrix(PCA_DIM)
W2 = ops.random_orthogonal_matrix(PCA_DIM)
X_source = X_source @ W1
X_target = X_target @ W2
X_test   = X_test   @ W2

source_rep = ops.MultipleLinear(PCA_DIM, LATENT_DIM, N_INSTANCES, bias=False).cuda()
target_rep = ops.MultipleLinear(PCA_DIM, LATENT_DIM, N_INSTANCES, bias=False).cuda()
classifier = ops.MultipleLinear(LATENT_DIM, 10, N_INSTANCES).cuda()

opt = optim.Adam((
        list(source_rep.parameters()) + 
        list(target_rep.parameters()) + 
        list(classifier.parameters())
    ), lr=LR)

def divergence_fn(X, Y):
    """Energy distance (biased)"""
    DXX = torch.cdist(X, X).mean(dim=[1,2])
    DYY = torch.cdist(Y, Y).mean(dim=[1,2])
    DXY = torch.cdist(X, Y).mean(dim=[1,2])
    return (DXY - DXX + DXY - DYY)

def forward():
    xs, ys = ops.get_batch([X_source, y_source], N_INSTANCES*BATCH_SIZE)
    xt = ops.get_batch(X_target, N_INSTANCES*BATCH_SIZE)
    xs = xs.view(N_INSTANCES, BATCH_SIZE, xs.shape[1])
    xt = xt.view(N_INSTANCES, BATCH_SIZE, xt.shape[1])
    ys = ys.view(N_INSTANCES, BATCH_SIZE)

    phi_s = source_rep(xs)
    phi_t = target_rep(xt)

    sinkhorn_loss = divergence_fn(phi_s, phi_t)
    classifier_loss = F.cross_entropy(classifier(phi_s).permute(0,2,1), ys,
        reduction='none').mean(dim=1)
    return classifier_loss, sinkhorn_loss

def calculate_test_accs():
    with torch.no_grad():
        X = torch.stack(N_INSTANCES*[X_test], dim=0)
        y = torch.stack(N_INSTANCES*[y_test], dim=0)
        logits = classifier(target_rep(X))
        return ops.multiclass_accuracy(logits, y).mean(dim=1)

scaler = torch.cuda.amp.GradScaler()

utils.print_row(
    'step',
    'classifier',
    'sinkhorn',
    'min test acc',
    'median test acc',
    'max test acc'
)

classifier_losses = []
sinkhorn_losses = []
for step in range(20001):
    utils.enforce_orthogonality(source_rep.weight)
    if TARGET_ORTHO:
        utils.enforce_orthogonality(target_rep.weight)

    with torch.cuda.amp.autocast():
        classifier_loss, sinkhorn_loss = [l.mean() for l in forward()]
        loss = (classifier_loss + (LAMBDA * sinkhorn_loss))
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    classifier_losses.append(classifier_loss.item())
    sinkhorn_losses.append(sinkhorn_loss.item())

    if step % 500 == 0:
        test_accs = calculate_test_accs()
        utils.print_row(
            step,
            np.mean(classifier_losses),
            np.mean(sinkhorn_losses),
            test_accs.min().item(),
            test_accs.median().item(),
            test_accs.max().item()
        )
        classifier_losses, sinkhorn_losses = [], []

with torch.no_grad():
    print('Final losses and test accs:')
    losses = torch.stack([forward()[1].detach() for _ in range(100)],
        dim=0).mean(dim=0)
    test_accs = calculate_test_accs()
    loss_argsort = torch.argsort(losses, descending=True)
    for idx in loss_argsort:
        utils.print_row(idx, losses[idx].item(), test_accs[idx].item())