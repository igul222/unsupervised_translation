"""
Q: Does ICA on MNIST yield latent variables with asymmetric distributions?

A: No, by visual inspection of most_symmetric_marginal.png
"""

import numpy as np
from sklearn.decomposition import FastICA
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib import utils, ops, datasets

N_COMPONENTS = 64

X, _ = datasets.mnist()

Z = torch.tensor(
    FastICA(n_components=N_COMPONENTS, max_iter=1000).fit_transform(X.cpu())
)

best_w_dist = np.inf
best_w_dist_z = None
for i in range(N_COMPONENTS):
    w_dist = ops.wasserstein_1d(Z[:, i], -Z[:, i])
    if w_dist < best_w_dist:
        best_w_dist = w_dist
        best_w_dist_z = Z[:, i]
    print(i, w_dist.item())

print('Smallest W-dist:', best_w_dist)
plt.hist(best_w_dist_z.cpu().numpy(), bins=100)
plt.savefig('most_symmetric_marginal.png')