"""
Is ICA approximately invariant to its own random seed, up to permutation/flips,
and can we recover the alignment using the component vectors?
Answer: Yes!
"""

import numpy as np
from sklearn.decomposition import FastICA
from torchvision import datasets
import lib
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
import os
import sys

N_COMPONENTS = 64
OUTPUT_DIR = 'outputs/04_mnist_ica_seed_invariance'

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.stdout = lib.Tee(f'{OUTPUT_DIR}/output.txt')

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

components_X = ica_X.components_
components_X2 = ica_X2.components_

# This block of code is a huge mess (TODO: rewrite), but basically
# it matches each component of ica_X to the closest component of ica_X2 by
# L2 distance (taking into account flips as well).
remaining_components = [c for c in components_X2]
matches = []
permutation = np.zeros((N_COMPONENTS, N_COMPONENTS))
for comp_i, comp in enumerate(components_X):
    min_dist = np.inf
    best_comp = None
    flipped = False
    for i, comp2 in enumerate(remaining_components):
        dist = ((comp - comp2)**2).sum()
        if dist < min_dist:
            min_dist = dist
            best_comp = i
            flipped = False
        dist = ((comp + comp2)**2).sum()
        if dist < min_dist:
            min_dist = dist
            best_comp = i
            flipped = True
    best_comp_i = [i for i,c2 in enumerate(components_X2)
        if np.array_equal(c2, remaining_components[best_comp])][0]
    if flipped:
        matches.append(-np.copy(remaining_components[best_comp]))
        permutation[best_comp_i, comp_i] = -1
    else:
        matches.append(np.copy(remaining_components[best_comp]))
        permutation[best_comp_i, comp_i] = 1

Z = ica_X.transform(X[:100])
Z = np.dot(Z, permutation.T)
Xhat = ica_X2.inverse_transform(Z)
lib.save_image_grid_mnist(Xhat + X_mean, f'{OUTPUT_DIR}/x_translated.png')

Z = ica_X.transform(X[:100])
Z = np.dot(Z, np.eye(N_COMPONENTS))
Xhat = ica_X.inverse_transform(Z)
lib.save_image_grid_mnist(Xhat + X_mean, f'{OUTPUT_DIR}/x_original.png')

def save_components(x, name):
    x = np.copy(x)
    x -= x.min()
    x /= x.max()
    lib.save_image_grid_mnist(x, name)
save_components(components_X, f'{OUTPUT_DIR}/components_X.png')
save_components(np.array(matches), f'{OUTPUT_DIR}/components_X2_matched.png')