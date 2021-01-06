"""
We attempt to find a linear translation of Colored MNIST by gradient descent on
energy distance.

Result: It doesn't work.
"""

import lib
import numpy as np
import torch
from torch import nn, optim

X_source, _, X_target, _ = lib.datasets.colored_mnist()

translation = nn.Linear(2*784, 2*784, bias=False).cuda()
opt = optim.Adam(translation.parameters(), lr=3e-4)

def forward():
    X_translated = translation(X_source)
    return lib.energy_dist.energy_dist(X_translated, X_target)

scaler = torch.cuda.amp.GradScaler()
lib.utils.print_row('step', 'loss')
for step in range(10001):
    loss = forward()
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    if step % 100 == 0:
        lib.utils.print_row(step, loss)
        lib.utils.save_image_grid(
            X_source[:100],
            f'step{str(step).zfill(5)}_original.png')
        lib.utils.save_image_grid(
            translation(X_source[:100]),
            f'step{str(step).zfill(5)}_translated.png')