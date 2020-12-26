"""
Q: Are the representations learned by ICA approximately invariant to
finite-sample noise, orthogonal transformations, and ICA's own random seed,
up to permutations / flips?

A: Yes. We can recover a flip/permutation matrix which translates between
two ICA representations of the same underlying data. The translation quality
is okay, but not great.
"""

import numpy as np
from sklearn.decomposition import FastICA
import torch
from lib import utils, ops, datasets, pca

N_COMPONENTS = 64

np.set_printoptions(suppress=True)

X, _ = datasets.mnist()
X1, X2 = X[::2], X[1::2]

# Whiten
pca1 = pca.PCA(X1, N_COMPONENTS, whiten=True)
pca2 = pca.PCA(X2, N_COMPONENTS, whiten=True)
X1 = pca1.forward(X1)
X2 = pca2.forward(X2)

# Apply random orthogonal transforms
W1 = ops.random_orthogonal_matrix(X1.shape[1])
W2 = ops.random_orthogonal_matrix(X2.shape[1])
X1 = X1 @ W1.T
X2 = X2 @ W2.T

ica1 = FastICA(max_iter=200, tol=1e-4, whiten=False).fit(X1.cpu())
ica2 = FastICA(max_iter=200, tol=1e-4, whiten=False).fit(X2.cpu())

# W takes z1 to z2.
W = (
    ica2.components_ @
    W2.cpu().numpy() @
    pca2.components.T.cpu().numpy() @
    pca1.components.cpu().numpy() @
    W1.T.cpu().numpy() @
    np.linalg.pinv(ica1.components_)
)

# Test 1: Is W orthogonal?
print(
    'L2 error of (W.T @ W - I):',
    np.sqrt(((np.eye(N_COMPONENTS) - W.T @ W)**2).sum())
)

# Test 2: does W "look" like a flip/permutation matrix?
print('Largest magnitudes of entries in each row of W:')
for row in W:
    print(np.sort(np.abs(row))[::-1][:5])

# Greedily turn W into a permutation/flip matrix, use it to translate
# X1 -> Z1 -> Z2 -> X2, and save the resulting images.
W_perm = np.zeros((N_COMPONENTS, N_COMPONENTS))
for max_idx in np.argsort(np.abs(W).flatten())[::-1]:
    row, col = max_idx // N_COMPONENTS, max_idx % N_COMPONENTS
    if np.sum(W_perm[row,:]) == 0 and np.sum(W_perm[:,col]) == 0:
        W_perm[row, col] = np.sign(W[row, col])

Xt = X1[:100].cpu().numpy()
Xt = ica1.transform(Xt)
Xt = Xt @ W_perm.T
Xt = ica2.inverse_transform(Xt)
Xt = pca2.inverse(torch.tensor(Xt).cuda().float() @ W2)

utils.save_image_grid_mnist(pca1.inverse(X1[:100] @ W1), 'X_original.png')
utils.save_image_grid_mnist(Xt, 'X_translated.png')
print('Saved X_original.png and X_translated.png; they should look similar.')