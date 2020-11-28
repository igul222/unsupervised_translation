"""
Can we find a linear map which translates red MNIST to green MNIST by gradient
descent on MMD? Not really; we seem to get stuck in local minima.
"""

import numpy as np
import torch
from torch import nn, optim
from lib import datasets, ops, utils

BATCH_SIZE = 512

mnist_red, mnist_green = datasets.colored_mnist()

translation = nn.Linear(2*784, 2*784).cuda()
opt = optim.Adam(translation.parameters(), lr=5e-4)

def forward():
    x_red = ops.get_batch(mnist_red, BATCH_SIZE)
    x_green = ops.get_batch(mnist_green, BATCH_SIZE)
    x_translated = translation(x_red)
    return ops.mmd(ops.energy_kernel, x_green, x_translated)

loss_ema = 0.
for step in range(10001):
    loss = forward()
    opt.zero_grad()
    loss.backward()
    opt.step()
    loss_ema = (0.01*loss.item()) + (0.99*loss_ema)
    if step % 1000 == 0:
        utils.print_row(step, loss_ema)
        utils.save_image_grid_colored_mnist(
            mnist_red[:100],
            f'step{str(step).zfill(5)}_original.png')
        utils.save_image_grid_colored_mnist(
            translation(mnist_red[:100]),
            f'step{str(step).zfill(5)}_translated.png')