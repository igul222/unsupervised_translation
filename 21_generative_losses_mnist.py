"""
Evaluation of Sinkhorn divergence, energy distance, and a nearest-neighbor loss
as alternatives to adversarial training for distribution matching.

Result: they all work, but none particularly well. The neighbors loss is the
most promising of the three, but it probably won't work on whitened data (too
much noise).
"""

import torch
from torch import nn, optim, autograd
import time
import geomloss
from lib import ops, utils, datasets
import argparse
import numpy as np
import collections

Z_DIM = 256
DIM = 1024
LR = 3e-4
BATCH_SIZE = 1024

X, _ = datasets.mnist()

sinkhorn_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=1.0,
    backend='tensorized')

energy_loss = ops.fast_energy_dist

@torch.jit.script
def _fast_neighbors(X, Y):
    X = X.half()
    Y = Y.half()
    DXY = torch.cdist(X, Y)
    return torch.argmin(DXY, dim=1)

def neighbors_loss(X_real, X_fake):
    with torch.cuda.amp.autocast(enabled=False):
        neighbors = _fast_neighbors(X_real, X_fake)
    return (X - Y[neighbors]).norm(p=2, dim=1).mean()

losses = [
    ('energy', energy_loss),
    ('sinkhorn', sinkhorn_loss),
    ('neighbors', neighbors_loss)
]

for name, loss_fn in losses:
    print('-'*80)
    print(name)

    model = nn.Sequential(
        nn.Linear(Z_DIM, DIM),
        nn.ReLU(),
        nn.Linear(DIM, DIM),
        nn.ReLU(),
        nn.Linear(DIM, 784)
    ).cuda()
    opt = optim.Adam(model.parameters(), lr=LR)

    def forward():
        X_real = ops.get_batch([X], BATCH_SIZE)[0]
        Z = torch.randn((BATCH_SIZE, Z_DIM), device='cuda')
        X_fake = model(Z)
        return loss_fn(X_real, X_fake)

    scaler = torch.cuda.amp.GradScaler()
    histories = collections.defaultdict(lambda: [])
    z_samples = torch.randn((100, Z_DIM)).cuda()
    start_time = time.time()
    utils.print_row('step', 'step time', 'loss')
    for step in range(20001):
        with torch.cuda.amp.autocast():
            loss = forward()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        histories['loss'].append(loss.item())
        if step % 1000 == 0:
            utils.print_row(
                step,
                (time.time() - start_time) / (step+1),
                np.mean(histories['loss'])
            )
            histories.clear()
            with torch.no_grad():
                samples = model(z_samples)
            utils.save_image_grid_mnist(samples,
                f'samples_{name}_step{step}.png')