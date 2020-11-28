"""
Can we translate red MNIST to green MNIST by minimizing an adversarial loss
between distributions?
Answer: Yes! (With several random restarts)
"""

import numpy as np
import torch
from torch import nn, optim, autograd
from torchvision import datasets
import torch.nn.functional as F
from lib import utils, ops, datasets, wgan_translation

N_TRANSLATIONS = 10

red_mnist, green_mnist = datasets.colored_mnist()
translations, losses = wgan_translation.translate(red_mnist, green_mnist,
    N_TRANSLATIONS, n_steps=10001)

utils.save_image_grid_colored_mnist(red_mnist[:100], 'source.png')

for i in range(N_TRANSLATIONS):
    print(f'attempt {i}: loss {losses[i]}')
    translated = red_mnist[:100] @ translations[i]
    utils.save_image_grid_colored_mnist(translated, f'translation_{i}.png')