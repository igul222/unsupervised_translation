"""
Does ICA on MNIST yield latent variables with asymmetric distributions?
Answer: No, by visual inspection of most_symmetric_marginal.png
"""

import numpy as np
from sklearn.decomposition import FastICA
from torchvision import datasets
import lib
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

N_COMPONENTS = 64
OUTPUT_DIR = 'outputs/02_mnist_ica_marginals'

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.stdout = lib.Tee(f'{OUTPUT_DIR}/output.txt')

mnist = datasets.MNIST('/tmp', train=True, download=True)
rng_state = np.random.get_state()
np.random.shuffle(mnist.data.numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist.targets.numpy())
X = (mnist.data.reshape((60000, 784)).float() / 256.).numpy()

ica = FastICA(n_components=N_COMPONENTS)
ica.fit(X)
Z = torch.tensor(ica.transform(X))

best_w_dist = np.inf
best_w_dist_z = None
for i in range(N_COMPONENTS):
    w_dist = lib.wasserstein_1d(Z[:, i], -Z[:, i])
    if w_dist < best_w_dist:
        best_w_dist = w_dist
        best_w_dist_z = Z[:, i]
    print(i, w_dist.item())

print('Smallest W-dist:', best_w_dist)
plt.hist(best_w_dist_z, bins=100)
plt.savefig(f'{OUTPUT_DIR}/most_symmetric_marginal.png')