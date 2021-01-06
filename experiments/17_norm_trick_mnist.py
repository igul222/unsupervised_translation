"""
Recover a random orthogonal translation of whitened MNIST by fourth-powering
the norm of each point and then aligning the PCs.

Result: it works, but it doesn't scale well past ~8 principal components.
"""

import lib
import torch
import tqdm

N_PCA = 8

errors = []
for _ in tqdm.tqdm(range(100)):
    X, _ = lib.datasets.mnist()

    X = lib.pca.PCA(X, N_PCA, whiten=True).forward(X)

    T_groundtruth = lib.ops.random_orthogonal_matrix(N_PCA)
    X_source = X[::2]
    X_target = X[1::2] @ T_groundtruth.T

    def transform(X):
        return X * X.norm(p=2, dim=1, keepdim=True).pow(3)

    X_source_transformed = transform(X_source)
    X_target_transformed = transform(X_target)

    source_pca = lib.pca.PCA(X_source_transformed, N_PCA, whiten=True)
    target_pca = lib.pca.PCA(X_target_transformed, N_PCA, whiten=True)
    As = source_pca.components[:X_source.shape[1], :]
    At = target_pca.components[:X_source.shape[1], :]

    # Fix flipped PCs by comparing the marginal distributions
    X_source_pca = source_pca.forward(X_source_transformed)
    X_target_pca = target_pca.forward(X_target_transformed)
    for i in range(N_PCA):
        d_unflipped = lib.ops.wasserstein_1d(X_source_pca[:, i], X_target_pca[:, i])
        d_flipped = lib.ops.wasserstein_1d(X_source_pca[:, i], -X_target_pca[:, i])
        if d_flipped < d_unflipped:
            At[:, i] *= -1

    # we should have At = T As
    T_hat = At @ torch.pinverse(As)

    error = torch.sqrt((T_hat - T_groundtruth).pow(2).sum())
    errors.append(error)

print('mean error', torch.mean(torch.stack(errors)).item())