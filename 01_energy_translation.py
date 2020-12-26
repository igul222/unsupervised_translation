"""
We attempt to find a linear translation of Colored MNIST by gradient descent on
energy distance.

Result: It doesn't work.
"""

import numpy as np
import torch
from torch import nn, optim
from lib import datasets, ops, utils

X_source, _, X_target, _ = datasets.colored_mnist()

translation = nn.Linear(2*784, 2*784, bias=False).cuda()
opt = optim.Adam(translation.parameters(), lr=3e-4)

def forward():
    X_translated = translation(X_source)
    return ops.fast_energy_dist(X_translated, X_target)

scaler = torch.cuda.amp.GradScaler()
utils.print_row('step', 'loss')
for step in range(10001):
    loss = forward()
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    if step % 100 == 0:
        utils.print_row(step, loss)
        utils.save_image_grid_colored_mnist(
            X_source[:100],
            f'step{str(step).zfill(5)}_original.png')
        utils.save_image_grid_colored_mnist(
            translation(X_source[:100]),
            f'step{str(step).zfill(5)}_translated.png')