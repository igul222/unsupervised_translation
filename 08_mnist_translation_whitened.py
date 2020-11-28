"""
Like the last experiment, but this time we whiten as a preprocessing step
and soft-constrain the maps to be orthogonal. Seems to work a little better.
"""

import numpy as np
import torch
from torch import nn, optim, autograd
from torchvision import datasets
import torch.nn.functional as F
from lib import utils, ops, datasets, wgan_translation, pca

N_TRANSLATIONS = 10
N_PCA = 64
ORTH_PENALTY = 0.01

red_mnist, green_mnist = datasets.colored_mnist()

red_pca = pca.PCA(red_mnist, N_PCA, whiten=True)
green_pca = pca.PCA(green_mnist, N_PCA, whiten=True)
red_mnist = red_pca.forward(red_mnist)
green_mnist = green_pca.forward(green_mnist)

translations, losses = wgan_translation.translate(red_mnist, green_mnist,
    N_TRANSLATIONS, orth_penalty=ORTH_PENALTY, n_steps=10001)

utils.save_image_grid_colored_mnist(red_pca.inverse(red_mnist[:100]),
    'source.png')

for i in range(N_TRANSLATIONS):
    print(f'attempt {i}: loss {losses[i]}')
    translated = green_pca.inverse(red_mnist[:100] @ translations[i])
    utils.save_image_grid_colored_mnist(translated, f'translation_{i}.png')