"""
IRM with a penalty forcing Z to have identity covariance, so that we can compute
the optimal least-squares predictor without inverting the covariance.

Status: Code works but no real conclusions.
"""
import argparse
import numpy as np
import torch
from torch import nn, optim, autograd
from lib import ops, utils, pca, datasets
import torch.nn.functional as F

INPUT_DIM = 2*196
PCA_DIM = 256
REPR_DIM = 128
LAMBDA_ERM = 0
LAMBDA_COV = 1

envs = datasets.irm_colored_mnist()

X_pooled = torch.cat([envs[0]['images'], envs[1]['images']], dim=0)
X_pca = pca.PCA(X_pooled, PCA_DIM, whiten=True)
X1, y1 = X_pca.forward(envs[0]['images']), envs[0]['labels']
X2, y2 = X_pca.forward(envs[1]['images']), envs[1]['labels']
X3, y3 = X_pca.forward(envs[2]['images']), envs[2]['labels']

# Apply random orthogonal transforms for optimization reasons.
W = ops.random_orthogonal_matrix(X1.shape[1])
X1 = X1 @ W.T
X2 = X2 @ W.T
X3 = X3 @ W.T

featurizer = nn.Linear(PCA_DIM, REPR_DIM, bias=False).cuda()
torch.nn.init.orthogonal_(featurizer.weight)
opt = optim.Adam(featurizer.parameters(), lr=1e-3)

def forward():
    Z1 = featurizer(X1)
    Z2 = featurizer(X2)

    y1_centered = (y1 - 0.5)
    y2_centered = (y2 - 0.5)

    phi1 = Z1.T @ y1_centered / Z1.shape[0]
    phi2 = Z2.T @ y2_centered / Z2.shape[0]

    preds1 = Z1 @ phi1
    preds2 = Z2 @ phi2

    irm_loss = (phi1 - phi2).pow(2).sum()

    train_mse = (
        (preds1 - y1_centered).pow(2).mean()
        + (preds2 - y2_centered).pow(2).mean()
    ) / 2.

    eye = torch.eye(Z1.shape[1], device='cuda')
    Z1_cov = (Z1.T @ Z1) / Z1.shape[0]
    Z2_cov = (Z2.T @ Z2) / Z2.shape[0]
    cov_penalty = (Z1_cov - eye).pow(2).sum() + (Z2_cov - eye).pow(2).sum()

    loss = (
        irm_loss
        + (LAMBDA_COV * cov_penalty)
        + (LAMBDA_ERM * train_mse) 
    )

    train_acc = (
        ops.binary_accuracy(preds1, y1)
        + ops.binary_accuracy(preds2, y1)
    ) / 2.

    phi_test = (phi1 + phi2) / 2.
    test_acc = ops.binary_accuracy(featurizer(X3) @ phi_test, y3)

    return loss, irm_loss, train_mse, cov_penalty, train_acc, test_acc

utils.print_row('step', 'loss', 'irm_loss', 'train_mse', 'cov_penalty',
    'train_acc', 'test_acc')
scaler = torch.cuda.amp.GradScaler()
for step in range(10001):
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        loss, irm_loss, train_mse, cov_penalty, train_acc, test_acc = forward()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    if step % 1000 == 0:
        utils.print_row(
            step,
            loss,
            irm_loss,
            train_mse,
            cov_penalty,
            train_acc,
            test_acc
        )