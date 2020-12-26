"""
Translation by minimizing a Sinkhorn divergence.
"""

import numpy as np
import torch
from torch import nn, optim
from lib import datasets, ops, utils
import geomloss

N_INSTANCES = 8
BATCH_SIZE = 1024
BLUR = 1.0

mnist_red, _, mnist_green, _ = datasets.colored_mnist()

loss_fn = geomloss.SamplesLoss(
    loss='sinkhorn',
    p=2,
    blur=BLUR,
    backend='tensorized',
)

translation = ops.MultipleLinear(2*784, 2*784, N_INSTANCES).cuda()
opt = optim.Adam(translation.parameters(), lr=5e-4)

def forward():
    x_red = ops.get_batch(mnist_red, BATCH_SIZE*N_INSTANCES, replacement=False)
    x_green = ops.get_batch(mnist_green, BATCH_SIZE*N_INSTANCES,
        replacement=False)
    x_red = x_red.view(N_INSTANCES, BATCH_SIZE, 2*784)
    x_green = x_green.view(N_INSTANCES, BATCH_SIZE, 2*784)
    x_translated = translation(x_red)
    return loss_fn(x_green, x_translated)

losses = []
for step in range(2001):
    loss = forward().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    utils.enforce_orthogonality(translation.weight)
    if step % 100 == 0:
        utils.print_row(step, np.mean(losses))
        losses = []
        with torch.no_grad():
            mnist_red_100 = torch.stack(N_INSTANCES*[mnist_red[:100]], dim=0)
            for inst in range(N_INSTANCES):
                utils.save_image_grid_colored_mnist(
                    mnist_red[:100],
                    f'step{str(step).zfill(5)}_instance{inst}_original.png')
                utils.save_image_grid_colored_mnist(
                    translation(mnist_red_100)[inst],
                    f'step{str(step).zfill(5)}_instance{inst}_translated.png')

print('Final losses:')
losses = torch.stack([forward().detach() for _ in range(100)], dim=0).mean(dim=0)
for i, loss in enumerate(losses):
    utils.print_row(i, loss.item())