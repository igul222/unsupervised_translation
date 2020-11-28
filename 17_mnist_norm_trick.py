"""
Recover a random orthogonal translation of whitened MNIST.
Method: transform the norm of each point, then align the PCs.
"""

import numpy as np
import torch
from torch import nn, optim, autograd
from lib import utils, ops, datasets, pca
import torch.nn.functional as F
import lib
import tqdm
from scipy.stats import ortho_group

N_PCA = 8

errors = []
for _ in tqdm.tqdm(range(100)):
    X, _ = datasets.mnist()

    X = pca.PCA(X, N_PCA, whiten=True).forward(X)

    T_groundtruth = torch.tensor(ortho_group.rvs(N_PCA)).float().cuda()
    X_source = X[::2]
    X_target = X[::2] @ T_groundtruth.T

    def raise_norm(X, power):
        return X * X.norm(p=2, dim=1, keepdim=True).pow(power - 1)

    X_source_transformed = raise_norm(X_source, 4)
    X_target_transformed = raise_norm(X_target, 4)

    source_pca = pca.PCA(X_source_transformed, N_PCA, whiten=True)
    target_pca = pca.PCA(X_target_transformed, N_PCA, whiten=True)
    As = source_pca.components
    At = target_pca.components

    # Fix flipped PCs by comparing the marginal distributions
    X_source_pca = source_pca.forward(X_source_transformed)
    X_target_pca = target_pca.forward(X_target_transformed)
    for i in range(N_PCA):
        d_unflipped = ops.wasserstein_1d(X_source_pca[:, i], X_target_pca[:, i])
        d_flipped = ops.wasserstein_1d(X_source_pca[:, i], -X_target_pca[:, i])
        if d_flipped < d_unflipped:
            At[:, i] *= -1

    # we should have At = T As
    T_hat = At @ torch.pinverse(As)

    error = torch.sqrt((T_hat - T_groundtruth).pow(2).sum())
    errors.append(error)

print('mean error', torch.mean(torch.stack(errors)).item())