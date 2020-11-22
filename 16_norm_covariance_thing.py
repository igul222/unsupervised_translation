"""
Whitened MNIST identity translation
"""

import numpy as np
import torch
from torch import nn, optim, autograd
from torchvision import datasets
import torch.nn.functional as F
import lib
from sklearn.decomposition import PCA
import os
import sys
from sklearn.decomposition import KernelPCA
import tqdm

PCA_DIMS = 8
OUTPUT_DIR = 'outputs/16_norm_covariance_thing'
# N_BUCKETS = 32
# N_KPCA_COMPONENTS = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.stdout = lib.Tee(f'{OUTPUT_DIR}/output.txt')

torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)

with torch.no_grad():

    mnist = datasets.MNIST('/tmp', train=True, download=True)
    rng_state = np.random.get_state()
    np.random.shuffle(mnist.data.numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist.targets.numpy())
    mnist_data = (mnist.data.reshape((60000, 784)).float() / 256.).clone().cuda()

    source = mnist_data[::2]
    target = mnist_data[1::2]

    def distance_matrix(X):
        BS = 1024
        M = torch.zeros((X.shape[0], X.shape[0])).cuda()
        for i in range(0, X.shape[0], BS):
            for j in range(0, X.shape[0], BS):
                X1 = X[i:i+BS]
                X2 = X[j:j+BS]
                block = torch.norm(X1[:, None, :] - X2[None, :, :], dim=2, p=2)
                M[i:i+BS, j:j+BS] = block
        return M

    def whiten(X):
        X = X.detach().cpu().numpy()
        pca = PCA(n_components=PCA_DIMS, whiten=True)
        return torch.tensor(pca.fit_transform(X)).float().cuda(), pca

    source, source_pca = whiten(source)
    target, target_pca = whiten(target)

    def sort_by_norm(X):
        norms = X.norm(p=2, dim=1)
        norms_argsort = torch.argsort(norms)
        return X[norms_argsort]

    def bucket_by_norm(X, n_buckets):
        X_sorted = sort_by_norm(X)
        bucket_size = X_sorted.shape[0] // n_buckets
        return [X_sorted[bucket_size*i:bucket_size*(i+1)]
            for i in range(n_buckets)]

    def neighbor_distance(X, k):
        M = distance_matrix(X)
        return torch.topk(M, k, dim=1, largest=False)[0].mean(dim=1)

    def bucket_by_neighbor_distance(X, k, n_buckets):
        dists = neighbor_distance(X, k)
        dists_argsort = torch.argsort(dists)
        X_sorted = X[dists_argsort]
        bucket_size = X_sorted.shape[0] // n_buckets
        return [X_sorted[bucket_size*i:bucket_size*(i+1)]
            for i in range(n_buckets)]
 
    def square_norms(X):
        # buckets = bucket_by_neighbor_distance(X, 10, 2)
        # buckets = bucket_by_norm(X, 2)
        # return torch.cat([buckets[0], buckets[1]], dim=0)
        return X * X.norm(p=2, dim=1, keepdim=True)
        # return buckets[-1]

    # print('source distance matrix')
    # Ms = distance_matrix(source)

    As = torch.tensor(whiten(square_norms(source))[1].components_.T).cuda()
    At = torch.tensor(whiten(square_norms(target))[1].components_.T).cuda()

    # source_buckets = bucket_by_neighbor_distance(source, 10, 20)
    # target_buckets = bucket_by_neighbor_distance(target, 10, 20)

    # source_means = [(b).mean(dim=0) for b in source_buckets]
    # target_means = [(b).mean(dim=0) for b in target_buckets]

    # As = torch.stack(source_means, dim=0).T
    # At = torch.stack(target_means, dim=0).T

    # As = sort_by_norm(source).T
    # At = sort_by_norm(target).T

    # print('source kpca')
    # source_kpca = KernelPCA(n_components=N_KPCA_COMPONENTS, kernel='rbf', fit_inverse_transform=True).fit(source.detach().cpu().numpy())
    # print('target kpca')
    # target_kpca = KernelPCA(n_components=N_KPCA_COMPONENTS, kernel='rbf', fit_inverse_transform=True).fit(target.detach().cpu().numpy())

    # # we should have At = T As
    T_hat = At @ torch.pinverse(As)
    print(T_hat)
    err = ((T_hat @ As) - At).pow(2).sum()
    print(err)

    import pdb; pdb.set_trace()

    # >>> X, _ = load_digits(return_X_y=True)
    # >>> transformer = KernelPCA(n_components=7, kernel='linear')
    # >>> X_transformed = transformer.fit_transform(X)
    # >>> X_transformed.shape
