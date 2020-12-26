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
REPR_DIM = 60
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


featurizer = nn.Linear(PCA_DIM, REPR_DIM, bias=False).cuda()
torch.nn.init.orthogonal_(featurizer.weight)
# classifier = nn.Sequential(
#     nn.Linear(REPR_DIM, 128), nn.ReLU(),
#     nn.Linear(128, 1)
# ).cuda()
classifier = nn.Linear(REPR_DIM, 1).cuda()

opt = optim.Adam([
    {'params': featurizer.parameters(), 'lr': LR_G},
    {'params': classifier.parameters(), 'lr': LR_G, 'weight_decay': 1e-1}
])

disc_targets = torch.zeros((X1.shape[0]+X2.shape[0], 1), device='cuda')
disc_targets[:X1.shape[0], :] += 1

eye = torch.eye(REPR_DIM, device='cuda')

def energy_dist(X, Y):
    DXX = torch.cdist(X, X).mean()
    DYY = torch.cdist(Y, Y).mean()
    DXY = torch.cdist(X, Y).mean()
    return (DXY - DXX) + (DXY - DYY)

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
    energy_loss = energy_dist(Z1_y, Z2_y) + energy_dist(Z1_not_y, Z2_not_y)

    loss = (
        energy_loss
        + (LAMBDA_NLL * nll)
    )

    acc = ops.binary_accuracy(logits1, y1) + ops.binary_accuracy(logits2, y2)
    acc = acc / 2.

    return loss, nll, energy_loss, acc

utils.print_row('step', 'loss', 'nll', 'energy_loss', 'acc', 'test acc')
scaler = torch.cuda.amp.GradScaler()
for step in range(10001):
    utils.enforce_orthogonality(featurizer.weight)

    with torch.cuda.amp.autocast():
        loss, nll, energy_loss, acc = forward()

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
            energy_loss,
            acc,
            test_acc
        )