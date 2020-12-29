"""
KPCA-translation on 2D Swiss Roll. We run KPCA on each domain to project points
into a common space, match points in this space, and then fit a translation
matrix from the matching. Note that we actually do PCA on quadratic features
rather than going through the kernel matrix.

Result: It works, but you need large sample size and low dataset noise. I
don't think this will ever scale beyond toy examples.
"""

import torch
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.decomposition
from scipy.stats import ortho_group

from lib import ops, pca, utils

N = 50000
PLOTS = False
N_KEY_POINTS = 20
DATASET_NOISE = 0.05

np.set_printoptions(precision=5, suppress=True)

def plot(X, title):
    if not PLOTS:
        return
    plt.clf()
    plt.scatter(X[:,0].cpu().numpy(), X[:,1].cpu().numpy())
    plt.title(title)
    plt.show()

def covariance(X):
    return torch.einsum('nx,ny->xy', X, X) / X.shape[0]

def quadratic_feats(X):
    feats = []
    for i in range(X.shape[1]):
        feats.append(X[:,i] * X[:,i])
        for j in range(0, i):
            feats.append(X[:,i] * X[:,j] * float(np.sqrt(2)))
    return torch.stack(feats, dim=1)

X = sklearn.datasets.make_swiss_roll(N, noise=DATASET_NOISE)[0]
X = np.stack([X[:,0], X[:,2]], axis=1)
np.random.shuffle(X)
X = torch.tensor(X).cuda()
Xs, Xt = X[::2], X[1::2]

Xs -= Xs.mean(dim=0, keepdim=True)
Xt -= Xt.mean(dim=0, keepdim=True)

# Random invertible matrix with eigenvalues 2, 1
T = ortho_group.rvs(2) @ np.array([[2., 0.], [0., 1.]]) @ ortho_group.rvs(2)
T = torch.tensor(T).cuda()
Xt = Xt @ T.T

plot(Xs, 'Xs')
plot(Xt, 'Xt')

pca_Xs = pca.PCA(Xs, 2, whiten=True)
pca_Xt = pca.PCA(Xt, 2, whiten=True)
Xs_white = pca_Xs.forward(Xs)
Xt_white = pca_Xt.forward(Xt)

Xs_feat = quadratic_feats(Xs_white)
Xt_feat = quadratic_feats(Xt_white)

Xs_feat_pca = pca.PCA(Xs_feat, Xs_feat.shape[1], whiten=True)
Xt_feat_pca = pca.PCA(Xt_feat, Xt_feat.shape[1], whiten=True)
Xs_feat_white = Xs_feat_pca.forward(Xs_feat)
Xt_feat_white = Xt_feat_pca.forward(Xt_feat)

for i in range(Xs_feat_white.shape[1]):
    p = Xs_feat_white[:, i]
    q = Xt_feat_white[:, i]
    unflipped_dist = ops.wasserstein_1d(p, q)
    flipped_dist = ops.wasserstein_1d(p, -q)
    if flipped_dist < unflipped_dist:
        p *= -1

D = ops.distance_matrix(Xs_feat_white, Xt_feat_white)
dists, nearest_neighbors = torch.min(D, dim=1)
best_matches = torch.argsort(dists, dim=0)
Xs_white_aligned = Xs_white[best_matches[:N_KEY_POINTS]]
Xt_white_aligned = Xt_white[nearest_neighbors[best_matches[:N_KEY_POINTS]]]

# We should have Xt_white_aligned.T = T_hat Xs_white_aligned.T
T_hat = Xt_white_aligned.T @ torch.pinverse(Xs_white_aligned.T)

# Compose the source-PCA -> T_hat -> inverse-target-PCA transformations
T_hat = (
    pca_Xt.components
    @ torch.diag(pca_Xt.magnitudes)
    @ T_hat
    @ torch.diag(1/pca_Xs.magnitudes)
    @ pca_Xs.components.T
)

print('T^-1 @ T_hat (should be identity):')
print(torch.inverse(T) @ T_hat)