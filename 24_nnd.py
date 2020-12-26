"""
Evaluation of Sinkhorn divergence as an alternative to adversarial training
for generative modeling.
"""

import torch
from torch import nn, optim, autograd
import time
import geomloss
from lib import ops, utils, datasets
import argparse
import numpy as np
import torch.nn.functional as F

Z_DIM = 256
DIM = 1024
LR = 3e-4
BATCH_SIZE = 1024
BLUR = 1.0

X, _ = datasets.mnist()
X = X.cuda()

model = nn.Sequential(
    nn.Linear(Z_DIM, DIM),
    nn.ReLU(),
    nn.Linear(DIM, DIM),
    nn.ReLU(),
    nn.Linear(DIM, 784),
    nn.Sigmoid()
).cuda()
opt = optim.Adam(model.parameters(), lr=LR)

def disc_loss(disc, X1, X2):
    return (
        F.binary_cross_entropy_with_logits(
            disc(X1), torch.ones((BATCH_SIZE, 1), device='cuda'))
        + F.binary_cross_entropy_with_logits(
            disc(X2), torch.zeros((BATCH_SIZE, 1), device='cuda'))
    )

def train_disc():
    disc = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
    ).cuda()
    disc_opt = optim.Adam(model.parameters(), lr=1e-3)

    scaler = torch.cuda.amp.GradScaler()
    for step in range(100):
        with torch.cuda.amp.autocast():
            X1 = ops.get_batch(X, BATCH_SIZE)
            X2 = model(torch.randn((BATCH_SIZE, Z_DIM), device='cuda'))
            loss = disc_loss(disc, X1, X2)
        disc_opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(disc_opt)
        scaler.update()
    return disc

def forward():
    disc = train_disc()
    X1 = ops.get_batch(X, BATCH_SIZE, replacement=True)
    X2 = model(torch.randn((BATCH_SIZE, Z_DIM), device='cuda'))
    return disc(X1).mean() - disc(X2).mean()

for step in range(20001):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.cuda.amp.autocast():
        loss = forward()
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    if step % 10 == 0:
        utils.print_row(step, loss.item(), time.time() - start_time)
        with torch.no_grad():
            z = torch.randn((100, Z_DIM)).cuda()
            samples = model(z)
        utils.save_image_grid_mnist(samples, f'samples_step{step}.png')
        del z, samples