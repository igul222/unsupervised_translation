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
PCA_DIM = 16
REPR_DIM = 15
DISC_DIM = 512
LAMBDA_ERM = 0#1e-4

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

random_feats_network = nn.Sequential(
    nn.Linear(INPUT_DIM, 512),
    nn.GELU(),
    nn.Linear(512, 512),
    nn.GELU(),
    nn.Linear(512, 512)
).cuda()
def random_feats(X):
    return X
    # return random_feats_network(X).detach()


X_pooled = torch.cat([envs[0]['images'], envs[1]['images']], dim=0)
X_pooled = random_feats(X_pooled)

X_pca = pca.PCA(X_pooled, PCA_DIM, whiten=True)
X1, y1 = X_pca.forward(random_feats(envs[0]['images'])), envs[0]['labels']
X2, y2 = X_pca.forward(random_feats(envs[1]['images'])), envs[1]['labels']
X3, y3 = X_pca.forward(random_feats(envs[2]['images'])), envs[2]['labels']

# Apply random orthogonal transforms for optimization reasons.
W = ops.random_orthogonal_matrix(X1.shape[1])
X1 = X1 @ W.T
X2 = X2 @ W.T
X3 = X3 @ W.T

N_FEATS = 256
W_ = torch.randn((N_FEATS, REPR_DIM), device='cuda') * 0.1
b_ = torch.rand((1, N_FEATS), device='cuda') * float(2*np.pi)
def quadratic_feats(x):
    return x
    # return torch.cos(x @ W_.T + b_)
    # return torch.cat([
    #     x,
    #     x.pow(2)
    #     ], dim=1)
    # return (x[:, :, None] * x[:, None, :]).reshape(x.shape[0], -1).clone()

class Abs(nn.Module):
    def forward(self, x):
        return x.abs()        

# featurizer = nn.Sequential(
#     nn.Linear(PCA_DIM, 64), Abs(),
#     nn.Linear(64, REPR_DIM)
# ).cuda()
# torch.nn.init.orthogonal_(featurizer[0].weight)
# torch.nn.init.orthogonal_(featurizer[2].weight)
featurizer = nn.Linear(PCA_DIM, REPR_DIM, bias=False).cuda()
torch.nn.init.orthogonal_(featurizer.weight)

opt = optim.Adam(featurizer.parameters(), lr=1e-3)

def forward():
    Z1 = featurizer(X1)
    Z2 = featurizer(X2)

    Z1 = quadratic_feats(Z1)
    Z2 = quadratic_feats(Z2)

    y1_centered = (y1 - 0.5)
    y2_centered = (y2 - 0.5)

    eye = torch.eye(Z1.shape[1], device='cuda')
    phi1 = torch.inverse(Z1.T @ Z1 + (Z1.shape[0] * 1e-3 * eye)) @ Z1.T @ y1_centered# / Z1.shape[0]
    phi2 = torch.inverse(Z2.T @ Z2 + (Z2.shape[0] * 1e-3 * eye)) @ Z2.T @ y2_centered# / Z2.shape[0]

    preds1 = Z1 @ phi1
    preds2 = Z2 @ phi2

    train_mse = (
        (preds1 - y1_centered).pow(2).mean()
        + (preds2 - y2_centered).pow(2).mean()
    ) / 2.

    loss = (phi1 - phi2).pow(2).sum() + (LAMBDA_ERM * train_mse)

    train_acc = (
        ops.binary_accuracy(preds1, y1)
        + ops.binary_accuracy(preds2, y1)
    ) / 2.

    phi_test = (phi1 + phi2) / 2.
    test_acc = ops.binary_accuracy(quadratic_feats(featurizer(X3)) @ phi_test, y3)

    return loss, train_mse, train_acc, test_acc

utils.print_row('step', 'loss', 'train_mse', 'train_acc', 'test_acc')
scaler = torch.cuda.amp.GradScaler()
for step in range(10001):
    # utils.enforce_orthogonality(featurizer[0].weight)
    # utils.enforce_orthogonality(featurizer[2].weight)
    utils.enforce_orthogonality(featurizer.weight)

    # with torch.cuda.amp.autocast():
    loss,train_mse, train_acc, test_acc = forward()

    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    if step % 100 == 0:
        utils.print_row(
            step,
            loss,
            train_mse,
            train_acc,
            test_acc
        )