"""
101st attempt at making IRM work (C-DANN).
"""
import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from lib import ops, utils, pca
import torch.nn.functional as F

INPUT_DIM = 2*196
PCA_DIM = 64
REPR_DIM = 32
LR_G = 1e-3
LR_D = 1e-3
DISC_DIM = 512

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

X = torch.cat([X1, X2], dim=0)
y = torch.cat([y1, y2], dim=0)
e = torch.zeros_like(y)
e[:len(X1)] += 1
y_one = y.nonzero(as_tuple=True)[0]
y_zero = (1-y).nonzero(as_tuple=True)[0]

featurizer = nn.Linear(PCA_DIM, REPR_DIM, bias=False).cuda()
torch.nn.init.orthogonal_(featurizer.weight)
classifier = nn.Sequential(
    nn.Linear(REPR_DIM, 1)
).cuda()
disc_y = nn.Sequential(
    nn.Linear(REPR_DIM, DISC_DIM), nn.GELU(),
    nn.Linear(DISC_DIM, DISC_DIM), nn.GELU(),
    nn.Linear(DISC_DIM, 1)
).cuda()
disc_not_y = nn.Sequential(
    nn.Linear(REPR_DIM, DISC_DIM), nn.GELU(),
    nn.Linear(DISC_DIM, DISC_DIM), nn.GELU(),
    nn.Linear(DISC_DIM, 1)
).cuda()

gen_params = (
    list(featurizer.parameters()) +
    list(classifier.parameters())
)
disc_params = (
    list(disc_y.parameters()) +
    list(disc_not_y.parameters())
)
opt = optim.Adam([
    {'params': gen_params, 'lr': LR_G, 'betas': (0., 0.99)},
    {'params': disc_params, 'lr': LR_D, 'betas': (0., 0.99)}
])

def forward():
    Z = featurizer(X)
    logits = classifier(Z.detach())
    nll = F.binary_cross_entropy_with_logits(logits, y)

    Z_y = Z[y_one, :]
    e_y = e[y_one, :]
    Z_not_y = Z[y_zero, :]
    e_not_y = e[y_zero, :]

    # Grad reversal trick
    Z_y = (2*Z_y.detach()) - Z_y
    Z_not_y = (2*Z_not_y.detach()) - Z_not_y

    disc_out_y = disc_y(Z_y)
    disc_out_not_y = disc_not_y(Z_not_y)

    disc_loss = (
        F.binary_cross_entropy_with_logits(  disc_out_y,         e_y)
        + F.binary_cross_entropy_with_logits(disc_out_not_y, e_not_y)
    ) / 2.

    grad = autograd.grad(disc_out_y.sum(), [Z_y], create_graph=True)[0]
    grad_penalty = grad.square().sum(dim=1).mean()

    grad = autograd.grad(disc_out_not_y.sum(), [Z_not_y], create_graph=True)[0]
    grad_penalty = grad_penalty + grad.square().sum(dim=1).mean()

    loss = disc_loss + nll + (1.*grad_penalty)

    acc = ops.binary_accuracy(logits, y)

    return loss, nll, disc_loss, grad_penalty, acc

utils.print_row('step', 'loss', 'nll', 'disc_loss', 'grad_penalty', 'acc', 'test acc')
scaler = torch.cuda.amp.GradScaler()
for step in range(10001):
    utils.enforce_orthogonality(featurizer.weight)

    with torch.cuda.amp.autocast():
        loss, nll, disc_loss, grad_penalty, acc = forward()

    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    if step % 100 == 0:
        test_acc = ops.binary_accuracy(classifier(featurizer(X3)), y3)
        utils.print_row(
            step,
            loss,
            nll,
            disc_loss,
            grad_penalty,
            acc,
            test_acc
        )