"""
Q: Can we find a linear translation of Colored MNIST by minimizing an
adversarial loss between distributions, when the data is whitened?

A: Yes. This actually makes optimization a lot easier, because we can constrain
the map to be orthogonal.
"""

import numpy as np
import torch
from torch import nn, optim, autograd
from torchvision import datasets
import torch.nn.functional as F
from lib import utils, ops, datasets, adversarial_translation, pca

N_INSTANCES = 8
PCA_DIM = 128

X_source, _, X_target, _ = datasets.colored_mnist()

source_pca = pca.PCA(X_source, PCA_DIM, whiten=True)
target_pca = pca.PCA(X_target, PCA_DIM, whiten=True)
X_source = source_pca.forward(X_source)
X_target = target_pca.forward(X_target)

# Apply random orthogonal transforms.
W1 = ops.random_orthogonal_matrix(X_source.shape[1])
W2 = ops.random_orthogonal_matrix(X_target.shape[1])
X_source = X_source @ W1.T
X_target = X_target @ W2.T

translations, divergences = adversarial_translation.train(
    X_source, X_target, N_INSTANCES, 
    disc_dim=512,
    lambda_gp=1.0,
    lambda_orth=0.1,
    lr_g=1e-3,
    lr_d=1e-3,
    print_freq=1000,
    steps=20001,
    weight_decay_d=1e-3
)

utils.save_image_grid_colored_mnist(
    source_pca.inverse(X_source[:100] @ W1), 'source.png')

for i in range(N_INSTANCES):
    translated = target_pca.inverse((X_source[:100] @ translations[i].T) @ W2)
    utils.save_image_grid_colored_mnist(translated, f'translation_{i}.png')