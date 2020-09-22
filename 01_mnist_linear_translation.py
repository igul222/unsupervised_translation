"""
Can we find a linear map which translates red MNIST to green MNIST by naive
gradient descent on MMD?
Answer: Not really; we get stuck in local minima.
"""

import numpy as np
import torch
from torch import nn, optim
import lib

BATCH_SIZE = 512

mnist_red_tr, mnist_green_tr, mnist_red_va, mnist_green_va = \
    lib.make_red_and_green_mnist()

translation = nn.Linear(2*196, 2*196).cuda()
opt = optim.Adam(translation.parameters(), lr=1e-3)

def forward():
    x_red = lib.get_batch(mnist_red_tr, BATCH_SIZE)
    x_green = lib.get_batch(mnist_green_tr, BATCH_SIZE)
    x_green_fake = translation(x_red)
    return lib.mmd(lib.energy_kernel, x_green, x_green_fake)

loss_ema = 0.
for step in range(10*1000):
    loss = forward()
    opt.zero_grad()
    loss.backward()
    opt.step()
    loss_ema = (0.05*loss.item()) + (0.95*loss_ema)
    if step % 100 == 0:
        lib.print_row(step, loss_ema)
        lib.save_image_grid_colored_mnist(
            mnist_red_tr[:100].cpu().detach().numpy(),
            f'step{str(step).zfill(5)}_original.png')
        lib.save_image_grid_colored_mnist(
            translation(mnist_red_tr[:100]).cpu().detach().numpy(),
            f'step{str(step).zfill(5)}_translated.png')