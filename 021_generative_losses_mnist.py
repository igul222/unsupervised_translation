"""
Evaluation of Sinkhorn divergence, energy distance, and a nearest-neighbor loss
as alternatives to adversarial training for distribution matching.

Result: they all work, but none particularly well. The neighbors loss is the
most promising of the three, but it probably won't work on whitened data (too
much noise).
"""

import functools
import geomloss
import lib
import torch
from torch import nn, optim

Z_DIM = 256
DIM = 1024
LR = 1e-3
BATCH_SIZE = 4096

X, _ = lib.datasets.mnist()

sinkhorn_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=1.0,
    backend='tensorized')

@torch.jit.script
def _fast_energy_dist(X, Y):
    X = X.half()
    Y = Y.half()
    DXX = torch.cdist(X, X)
    DXY = torch.cdist(X, Y)
    DYY = torch.cdist(Y, Y)
    M = (DXY - DXX + DXY - DYY).float()
    N = M.shape[0]
    arange_N = torch.arange(N, device='cuda')
    mu = M.mean()
    mu_diag = M[arange_N, arange_N].mean()
    return ((N / (N-1)) * mu) - (mu_diag / (N-1))

def energy_loss(X, Y):
    with torch.cuda.amp.autocast(enabled=False):
        return _fast_energy_dist(X, Y)

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

@torch.jit.script
def _fast_mmd(X, Y, gamma: float):
    X = X.half()
    Y = Y.half()
    KXX = torch.exp(-gamma*torch.cdist(X, X))
    KXY = torch.exp(-gamma*torch.cdist(X, Y))
    KYY = torch.exp(-gamma*torch.cdist(Y, Y))
    M = (KXX - KXY + KYY - KXY).float()
    N = M.shape[0]
    arange_N = torch.arange(N, device='cuda')
    mu = M.mean()
    mu_diag = M[arange_N, arange_N].mean()
    return ((N / (N-1)) * mu) - (mu_diag / (N-1))

def mmd(X, Y, gamma):
    with torch.cuda.amp.autocast(enabled=False):
        return _fast_mmd(X, Y, gamma=gamma)

losses = [
    ('mmd', functools.partial(mmd, gamma=1e-1)),
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
        X_real = lib.ops.get_batch([X], BATCH_SIZE)[0]
        Z = torch.randn((BATCH_SIZE, Z_DIM), device='cuda')
        X_fake = model(Z)
        return loss_fn(X_real, X_fake)

    lib.utils.train_loop(forward, opt, 10001)

    samples = model(torch.randn((100, Z_DIM), device='cuda'))
    lib.utils.save_image_grid(samples, f'samples_{name}.png')