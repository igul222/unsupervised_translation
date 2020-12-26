"""
Covariance of quadratic features
"""

import numpy as np
import torch
from torch import nn, optim, autograd
from lib import ops, utils, pca, datasets
import torch.nn.functional as F

PCA_DIM = 2

X, y = datasets.mnist()
X_pca = pca.PCA(X, PCA_DIM, whiten=True)
X = X_pca.forward(X)

def cov(x):
    return torch.einsum('nx,ny->nxy', x, x).mean(dim=0)

print(X.mean(dim=0))
print(cov(X))

def quadratic_feats(x):
    feats = []
    for i in range(x.shape[1]):
        feats.append(x[:,i])
    for i in range(x.shape[1]):
        for j in range(i+1, x.shape[1]):
            feats.append(x[:,i] * x[:,j])
    return torch.stack(feats, dim=1)

X_quad = quadratic_feats(X)
print(cov(X_quad))