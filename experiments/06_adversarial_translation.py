"""
Q: Can we find a linear translation of Colored MNIST by minimizing an
adversarial loss between distributions?

A: Yes. Adversarial translation succeeds roughly 1/8 of the time (with these
hyperparameters), so we train 8 instances simultaneously and compute energy
distances to select the best model.
"""

import lib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd

N_INSTANCES = 8

X_source, _, X_target, _ = lib.datasets.colored_mnist()
translations, divergences = lib.adversarial.train_translation(
    X_source, X_target, N_INSTANCES, 
    lambda_gp=3.0,
    lambda_orth=0.,
    disc_dim=512,
    lr_g=5e-5,
    lr_d=2e-4,
    print_freq=1000,
    steps=30001,
    l2reg_d=1e-3
)

lib.utils.save_image_grid(X_source[:100], 'source.png')

for i in range(N_INSTANCES):
    translated = X_source[:100] @ translations[i].T
    lib.utils.save_image_grid(translated, f'translation_{i}.png')