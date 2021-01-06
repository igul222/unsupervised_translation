"""
Demonstration that spurious invariances exist in semi-realistic settings: we
train a supervised model, extract the logit distribution, and then find
a projection of the target which matches that distribution.

Result: The learned target classifier matches the distribution well, but
doesn't attain better than random-chance classification accuracy.
"""

import lib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim

PCA_DIM = 128
STEPS = 3001

X_source, y_source, X_target, y_target = lib.datasets.binary_colored_mnist()
source_pca = lib.pca.PCA(X_source, PCA_DIM, whiten=True)
target_pca = lib.pca.PCA(X_target, PCA_DIM, whiten=True)
X_source = source_pca.forward(X_source)
X_target = target_pca.forward(X_target)
# Apply random orthogonal transforms for optimization reasons.
W1 = lib.ops.random_orthogonal_matrix(X_source.shape[1])
W2 = lib.ops.random_orthogonal_matrix(X_target.shape[1])
X_source = X_source @ W1.T
X_target = X_target @ W2.T

W_source = nn.Linear(PCA_DIM, 2).cuda()
def forward():
    Z_source = W_source(X_source)
    return F.cross_entropy(Z_source, y_source)
    source_acc = lib.ops.multiclass_accuracy(Z_source, y_source).mean()

opt = optim.Adam(W_source.parameters())
lib.utils.train_loop(forward, opt, 3001)
Z_source = W_source(X_source).detach()

def plot(name, Z_source, Z_target):
    """Save plots of Z_source and Z_target"""
    def to_np(tensor):
        return tensor.cpu().detach().numpy()
    plt.clf()
    plt.scatter(to_np(Z_source[:,0]), to_np(Z_source[:,1]),
        c=to_np(y_source), cmap='tab10')
    plt.savefig(f'{name}_source.png')
    plt.clf()
    plt.scatter(to_np(Z_target[:,0]), to_np(Z_target[:,1]),
        c=to_np(y_target), cmap='tab10')
    plt.savefig(f'{name}_target.png')

W_target = nn.Linear(PCA_DIM, 2).cuda()
def forward():
    Z_target = W_target(X_target)
    target_acc = lib.ops.multiclass_accuracy(Z_target, y_target).mean()
    return lib.energy_dist.energy_dist(Z_source, Z_target), target_acc
opt = optim.Adam(W_target.parameters(), lr=1e-2)
lib.utils.train_loop(forward, opt, STEPS, history_names=['target_acc'],
    print_freq=100)
Z_target = W_target(X_target).detach()
plot('random', Z_source, Z_target)

W_target = nn.Linear(PCA_DIM, 2).cuda()
def forward():
    Z_target = W_target(X_target)
    target_acc = lib.ops.multiclass_accuracy(Z_target, y_target).mean()
    energy_dist = lib.ops.fast_energy_dist(Z_source, Z_target)
    cross_entropy = F.cross_entropy(Z_target, 1-y_target)
    loss = energy_dist + (0.01*cross_entropy)
    return loss, energy_dist, cross_entropy, target_acc
opt = optim.Adam(W_target.parameters(), lr=1e-2)
lib.utils.train_loop(forward, opt, STEPS, history_names=['energy_dist',
    'cross_entropy', 'target_acc'], print_freq=100)
Z_target = W_target(X_target).detach()
plot('worstcase', Z_source, Z_target)
