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

Z_DIM = 256
DIM = 1024
LR = 3e-4
BATCH_SIZE = 4096
BLUR = 1.0

X, _ = datasets.mnist()
X = X.cuda()
X_mean = X.mean(dim=0, keepdim=True)
X -= X_mean

model = nn.Sequential(
    nn.Linear(Z_DIM, DIM),
    nn.ReLU(),
    nn.Linear(DIM, DIM),
    nn.ReLU(),
    nn.Linear(DIM, 784)
).cuda()

opt = optim.Adam(model.parameters(), lr=LR)

def fast_pairwise_distances(x, y):
    x_norm = x.pow(2).sum(dim=1)[:,None]
    y_norm = y.pow(2).sum(dim=1)[None,:]
    x_dot_y = torch.einsum('ab,cb->ac', x, y)
    dist = (x_norm - x_dot_y) + (y_norm - x_dot_y)
    return dist

def cdist_pairwise_distances(X, Y):
    return torch.cdist(X[None,:,:],Y[None,:,:])[0]

SIGMA = 10.0
GAMMA = 1 / (2 * SIGMA)
def gaussian_kernel(X, Y):
    return torch.exp(-GAMMA*(cdist_pairwise_distances(X, Y)**2))

def energy_kernel(X, Y):
    return -cdist_pairwise_distances(X, Y)

def poly_kernel(X, Y):
    x_dot_y = torch.einsum('an,bn->ab', X, Y)
    return torch.tanh(1.0 * x_dot_y)
    # return x_dot_y + x_dot_y.pow(2) + x_dot_y.pow(3)

_weights = {}
def matrix_mean(d, triu):
    if triu:
        if d.shape not in _weights:
            _weights[d.shape] = torch.triu(torch.ones_like(d), diagonal=1)
        weights = _weights[d.shape]
        return torch.mean(d * weights) / torch.mean(weights)
    else:
        return torch.mean(d)
 
def mmd(kernel_fn, x, y, triu_x=True, triu_y=True):
    return (-2*matrix_mean(kernel_fn(x, y), False)
        + matrix_mean(kernel_fn(x, x), triu_x)
        + matrix_mean(kernel_fn(y, y), triu_y))

def forward():
    X_batch = ops.get_batch(X, BATCH_SIZE, replacement=True)
    Z = torch.randn((BATCH_SIZE, Z_DIM), device='cuda')
    X_gen = model(Z)
    return mmd(energy_kernel, X_batch, X_gen, triu_x=False, triu_y=False)

scaler = torch.cuda.amp.GradScaler()

for step in range(20001):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.cuda.amp.autocast():
        loss = forward()
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize()
    if step % 100 == 0:
        # with torch.no_grad():
        #     X_batch = ops.get_batch(X, BATCH_SIZE, replacement=True)
        #     Z = torch.randn((BATCH_SIZE, Z_DIM), device='cuda')
        #     X_gen = model(Z)
        #     print('median heuristic real', fast_pairwise_distances(X_batch, X_batch).median())
        #     print('median heuristic fake', fast_pairwise_distances(X_gen, X_gen).median())
        #     print('median heuristic cross', fast_pairwise_distances(X_batch, X_gen).median())

        utils.print_row(step, loss.item(), time.time() - start_time)
        with torch.no_grad():
            z = torch.randn((100, Z_DIM)).cuda()
            samples = model(z) + X_mean
        utils.save_image_grid_mnist(samples, f'samples_step{step}.png')
        del z, samples