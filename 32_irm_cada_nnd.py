"""
102nd attempt at making IRM work (C-DANN + NND).
"""
import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from lib import ops, utils, pca
import torch.nn.functional as F
import copy

INPUT_DIM = 2*196
PCA_DIM = 64
REPR_DIM = 32
LAMBDA_NLL = 1
LR_G = 1e-3

mnist = datasets.MNIST('~/data', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])

rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())

# Build environments

def make_environment(images, labels, e):
    def torch_bernoulli(p, size):
      return (torch.rand(size) < p).float()
    def torch_xor(a, b):
      return (a-b).abs() # Assumes both inputs are either 0 or 1
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
      'images': (images.float() / 255.).view(images.shape[0], 2*196).cuda(),
      'labels': labels[:, None].cuda()
    }

envs = [
    make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.1),
    make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.2),
    make_environment(mnist_val[0], mnist_val[1], 0.9)
]

X_pooled = torch.cat([envs[0]['images'], envs[1]['images']], dim=0)
X_pca = pca.PCA(X_pooled, PCA_DIM, whiten=True)
X1, y1 = X_pca.forward(envs[0]['images']), envs[0]['labels']
X2, y2 = X_pca.forward(envs[1]['images']), envs[1]['labels']
X3, y3 = X_pca.forward(envs[2]['images']), envs[2]['labels']

# Apply random orthogonal transforms for optimization reasons.
W = ops.random_orthogonal_matrix(X1.shape[1])
X1 = X1 @ W.T
X2 = X2 @ W.T
X3 = X3 @ W.T

class NND:
    def __init__(self, dim_in, batch_size=64, n_steps=50):
        self.batch_size = batch_size
        self.n_steps = n_steps

        self.disc = nn.Sequential(
            nn.Linear(dim_in, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1)
        ).cuda()

        self.opt = optim.Adam(self.disc.parameters(), lr=1e-4)
        self.opt_initial_state_dict = copy.deepcopy(self.opt.state_dict())

    def evaluate(self, X1, X2):
        torch.nn.init.xavier_uniform_(self.disc[0].weight)
        torch.nn.init.xavier_uniform_(self.disc[2].weight)
        torch.nn.init.xavier_uniform_(self.disc[4].weight)
        torch.nn.init.zeros_(self.disc[0].bias)
        torch.nn.init.zeros_(self.disc[2].bias)
        torch.nn.init.zeros_(self.disc[4].bias)
        self.opt.load_state_dict(self.opt_initial_state_dict)

        X = torch.cat([X1.detach(), X2.detach()], dim=0)
        y = torch.zeros((X.shape[0], 1), device='cuda')
        y[:X1.shape[0], :] += 1

        scaler = torch.cuda.amp.GradScaler()

        steps = 0
        while True:
            perm = torch.randperm(len(X), device='cuda')
            X_shuf, y_shuf = X[perm], y[perm]
            for offset in range(0, len(X)-self.batch_size+1, self.batch_size):
                X_batch = X_shuf[offset:offset+self.batch_size].detach()
                y_batch = y_shuf[offset:offset+self.batch_size].detach()

                loss = F.binary_cross_entropy_with_logits(
                    self.disc(X_batch), y_batch)

                with torch.cuda.amp.autocast(enabled=False):
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                    self.opt.zero_grad(set_to_none=True)

                steps += 1
                if steps == self.n_steps:
                    break

            if steps == self.n_steps:
                break

        loss = (
            F.binary_cross_entropy_with_logits(self.disc(X1), y[:len(X1)])
            + F.binary_cross_entropy_with_logits(self.disc(X2), y[len(X1):])
        ) / 2.

        return (float(np.log(2)) - loss)

nnd1 = NND(REPR_DIM, batch_size=8192, n_steps=100)
nnd2 = NND(REPR_DIM, batch_size=8192, n_steps=100)
featurizer = nn.Linear(PCA_DIM, REPR_DIM, bias=False).cuda()
torch.nn.init.orthogonal_(featurizer.weight)
classifier = nn.Sequential(
    nn.Linear(REPR_DIM, 1)
).cuda()

gen_params = (
    list(featurizer.parameters()) +
    list(classifier.parameters())
)
opt = optim.Adam([
    {'params': gen_params, 'lr': LR_G},
])

eye = torch.eye(REPR_DIM, device='cuda')

def forward():
    Z1 = featurizer(X1)
    Z2 = featurizer(X2)

    logits1 = classifier(Z1.detach())
    logits2 = classifier(Z2.detach())

    nll = F.binary_cross_entropy_with_logits(logits1, y1)
    nll = nll + F.binary_cross_entropy_with_logits(logits2, y2)
    nll = nll / 2.

    Z1_y = Z1[y1.nonzero(as_tuple=True)[0], :]
    Z2_y = Z2[y2.nonzero(as_tuple=True)[0], :]
    Z1_not_y = Z1[(1-y1).nonzero(as_tuple=True)[0], :]
    Z2_not_y = Z2[(1-y2).nonzero(as_tuple=True)[0], :]
    nnd_loss = nnd1.evaluate(Z1_y, Z2_y) + nnd2.evaluate(Z1_not_y, Z2_not_y)

    loss = (
        nnd_loss
        + (LAMBDA_NLL * nll)
    )

    acc = ops.binary_accuracy(logits1, y1) + ops.binary_accuracy(logits2, y2)
    acc = acc / 2.

    return loss, nll, nnd_loss, acc

utils.print_row('step', 'loss', 'nll', 'nnd_loss', 'acc', 'test acc')
scaler = torch.cuda.amp.GradScaler()
for step in range(10001):
    utils.enforce_orthogonality(featurizer.weight)

    with torch.cuda.amp.autocast(enabled=True):
        loss, nll, nnd_loss, acc = forward()

    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    if step % 10 == 0:
        test_acc = ops.binary_accuracy(classifier(featurizer(X3)), y3)
        utils.print_row(
            step,
            loss,
            nll,
            nnd_loss,
            acc,
            test_acc
        )