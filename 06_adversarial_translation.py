"""
Q: Can we find a linear translation of Colored MNIST by minimizing an
adversarial loss between distributions?

A: Yes. Adversarial translation succeeds roughly 1/8 of the time (with these
hyperparameters), so we train 8 instances simultaneously and compute energy
distances to select the best model.
"""

import numpy as np
import torch
from torch import nn, optim, autograd
from torchvision import datasets
import torch.nn.functional as F
from lib import utils, ops, datasets, adversarial_translation

N_INSTANCES = 8

X_source, _, X_target, _ = datasets.colored_mnist()
translations, energy_dists = adversarial_translation.train(
    X_source, X_target, N_INSTANCES, 
    lambda_gp=3.0,
    lambda_orth=0.,
    disc_dim=512,
    lr_g=5e-5,
    lr_d=2e-4,
    print_freq=1000,
    steps=30001,
    weight_decay_d=1e-3
)

utils.save_image_grid_colored_mnist(X_source[:100], 'source.png')

for i in range(N_INSTANCES):
    translated = X_source[:100] @ translations[i].T
    utils.save_image_grid_colored_mnist(translated, f'translation_{i}.png')