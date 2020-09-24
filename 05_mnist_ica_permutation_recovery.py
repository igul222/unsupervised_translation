"""
ICA is ~invariant to its own random seed up to permutation/flips (as
demonstrated by the last experiment); how much of the permutation matrix can
we recover using only the distributions of the marginals in the latent space?
"""

import numpy as np
from sklearn.decomposition import FastICA
from torchvision import datasets
import lib
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from tqdm import tqdm

N_COMPONENTS = 64

mnist = datasets.MNIST('/tmp', train=True, download=True)
rng_state = np.random.get_state()
np.random.shuffle(mnist.data.numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist.targets.numpy())
X = (mnist.data.reshape((60000, 784)).float() / 256.).numpy()
X_mean = X.mean(axis=0)
X -= X_mean

ica_X = FastICA(n_components=N_COMPONENTS, max_iter=1000, random_state=1)
ica_X.fit(X)

ica_X2 = FastICA(n_components=N_COMPONENTS, max_iter=1000, random_state=2)
ica_X2.fit(X)

Z1 = torch.tensor(ica_X.transform(X))
Z2 = torch.tensor(ica_X2.transform(X))

Z1_sorted = torch.zeros_like(Z1)
Z2_sorted = torch.zeros_like(Z2)
for i in range(N_COMPONENTS):
    z1_vals, _ = torch.sort(Z1[:,i])
    z2_vals, _ = torch.sort(Z2[:,i])
    Z1_sorted[:,i] = z1_vals
    Z2_sorted[:,i] = z2_vals
Z2_flipped_sorted = torch.flip(-Z2_sorted, [0])

def wasserstein_1D_sorted(p, q):
    return torch.abs(p - q).mean()

torch.set_printoptions(sci_mode=False, precision=8)
permutation = np.zeros((N_COMPONENTS, N_COMPONENTS))
z1_z2_pairs = []
for z1_idx in tqdm(list(range(N_COMPONENTS))):
    fits = [] # each entry is a (w1_dist, z2_idx, flipped) tuple
    for z2_idx in range(N_COMPONENTS):
        w1_dist = wasserstein_1D_sorted(Z1_sorted[:,z1_idx], Z2_sorted[:,z2_idx])
        w1_dist_flipped = wasserstein_1D_sorted(Z1_sorted[:,z1_idx], Z2_flipped_sorted[:,z2_idx])
        fits.append((w1_dist, z2_idx, False))
        fits.append((w1_dist_flipped, z2_idx, True))
    fits = sorted(fits, key=lambda x: x[0])
    if 2 * fits[0][0] < fits[1][0]:
        print('Clear winner:', z1_idx, fits[0])
        _, z2_idx, flipped = fits[0]
        permutation[z2_idx, z1_idx] = (-1 if flipped else 1)
        z1_z2_pairs.append((z1_idx, z2_idx))

Z = ica_X.transform(X[:100])
Z = np.dot(Z, permutation.T)
Xhat = ica_X2.inverse_transform(Z)
lib.save_image_grid_mnist(Xhat + X_mean, 'mnist_ica_permutation_recovery_xhat_translated.png')

Z = ica_X.transform(X[:100])
Z = np.dot(Z, np.eye(N_COMPONENTS))
Xhat = ica_X.inverse_transform(Z)
lib.save_image_grid_mnist(Xhat + X_mean, 'mnist_ica_permutation_recovery_xhat_orig.png')

components_X = ica_X.components_
components_X2 = np.dot(ica_X2.components_.T, permutation).T

def save_components(x, name):
    x = np.copy(x)
    x -= x.min()
    x /= x.max()
    lib.save_image_grid_mnist(x, name)
save_components(components_X, 'mnist_ica_permutation_recovery_components_X.png')
save_components(components_X2, 'mnist_ica_permutation_recovery_components_X2_matched.png')