"""
Evaluation of random features as an alternative to adversarial training
for generative modeling.
"""

import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import time
import geomloss
from lib import ops, utils, datasets, pca
import argparse
import numpy as np

Z_DIM = 256
DIM = 512
LR = 1e-3
N_FEATS = 16384
PCA_DIM = 128
W_SCALE = 0.1

X, _ = datasets.mnist()
X_pca = pca.PCA(X, PCA_DIM)
X = X_pca.forward(X)

model = nn.Sequential(
    nn.Linear(Z_DIM, DIM),
    nn.ReLU(),
    nn.Linear(DIM, DIM),
    nn.ReLU(),
    nn.Linear(DIM, PCA_DIM)
).cuda()

opt = optim.Adam(model.parameters(), lr=LR)

rand_W = torch.randn((N_FEATS, X.shape[1]), device='cuda') * W_SCALE
rand_b = torch.rand(1,N_FEATS, device='cuda') * float(2*np.pi)
def random_feats(x):
    Wx = torch.addmm(rand_b, x, rand_W.T)
    return torch.cos(Wx)

with torch.no_grad():
    X_feats = random_feats(X).mean(dim=0).detach()

def forward():
    Z = torch.randn((X.shape[0], Z_DIM), device='cuda')
    X_gen = model(Z)
    X_gen_feats = random_feats(X_gen).mean(dim=0)
    return (X_feats - X_gen_feats).pow(2).mean()

scaler = torch.cuda.amp.GradScaler()

for step in range(20001):
    with torch.cuda.amp.autocast(enabled=True):
        loss = forward()
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    if step % 100 == 0:
        utils.print_row(step, loss.item())
        with torch.no_grad():
            z = torch.randn((100, Z_DIM)).cuda()
            samples = X_pca.inverse(model(z))
        utils.save_image_grid_mnist(samples, f'samples_step{step}.png')
        del z, samples