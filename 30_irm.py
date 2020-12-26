"""
100th attempt at making IRM work.
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
LAMBDA = 1e3

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
    make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
    make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
    make_environment(mnist_val[0], mnist_val[1], 0.9)
]

featurizer = nn.Linear(PCA_DIM, REPR_DIM, bias=False).cuda()
torch.nn.init.orthogonal_(featurizer.weight)
classifier = nn.Linear(REPR_DIM, 1, bias=False).cuda()

opt = optim.Adam(
    (
        list(featurizer.parameters())
        + list(classifier.parameters())
    ),
    lr=1e-3
)

X_pooled = torch.cat([envs[0]['images'], envs[1]['images']], dim=0)
X_pca = pca.PCA(X_pooled, PCA_DIM, whiten=True)
for env in envs:
    env['images'] = X_pca.forward(env['images'])

eye = torch.eye(REPR_DIM, device='cuda')

utils.print_row('step', 'train nll', 'train acc', 'train penalty', 'test acc')
for step in range(10001):
    utils.enforce_orthogonality(featurizer.weight)

    for env in envs:
        feats = featurizer(env['images'])
        ones = torch.ones((1, REPR_DIM), device='cuda')
        ones.requires_grad = True
        logits = classifier(feats * ones)
        env['nll'] = F.binary_cross_entropy_with_logits(logits, env['labels'])
        env['acc'] = ops.binary_accuracy(logits, env['labels'])
        grad = autograd.grad(env['nll'], [ones], create_graph=True)[0]
        env['penalty'] = grad.pow(2).sum()

    train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
    train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
    train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

    loss = train_nll.clone()
    loss += LAMBDA * train_penalty

    opt.zero_grad()
    loss.backward()
    opt.step()

    test_acc = envs[2]['acc']
    if step % 100 == 0:
      utils.print_row(
        step,
        train_nll,
        train_acc,
        train_penalty,
        test_acc
      )