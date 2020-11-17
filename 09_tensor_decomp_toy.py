"""
Translation by tensor decomposition on a 2D toy example.
"""

import numpy as np
import sklearn.decomposition
import tensorly as tl
import tensorly.decomposition
import lib
import os
import sys

N = 10*1000*1000
OUTPUT_DIR = 'outputs/09_tensor_decomp_toy'

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.stdout = lib.Tee(f'{OUTPUT_DIR}/output.txt')

np.set_printoptions(suppress=True, precision=8)

def half_parabola(x):
    return np.maximum(0, x)**2

Z = np.random.randn(N)
X_source = np.stack([Z, half_parabola(Z)], axis=1)

# Whiten X_source
X_source -= np.mean(X_source, axis=0)
X_source = sklearn.decomposition.PCA(whiten=True).fit_transform(X_source)

# Ground-truth orthogonal translation T
T_groundtruth = np.array([[0., 1.], [1., 0.]])
X_target = X_source @ T_groundtruth.T

# Independent samples in each distribution
X_source = X_source[::2]
X_target = X_target[1::2]

def third_moment(X):
    return np.einsum('nx,ny,nz->xyz', X, X, X) / X.shape[0]

Ms = third_moment(X_source)
Mt = third_moment(X_target)

def decomp(tensor):
    for rank in range(1, 10):
        (w, (A, B, C)), errors = tl.decomposition.parafac(tensor, rank=rank,
            return_errors=True, normalize_factors=True)
        print(f'decomp: rank {rank}, error {errors[-1]}')
        if errors[-1] < 1e-2:
            break
    # Canonicalize the decomposition
    w_sort = np.argsort(w)
    w = w[w_sort]
    A = A[:,w_sort]
    B = B[:,w_sort]
    C = C[:,w_sort]
    return w, A, B, C

print('PARAFAC:')

print('source:')
ws, As, Bs, Cs = decomp(Ms)
print('target:')
wt, At, Bt, Ct = decomp(Mt)

print('ws', ws)
print('wt', wt)
print('As', As)
print('At', At)

# We should have that At = T As
T_hat = At @ np.linalg.pinv(As)
print(T_hat)

print('Symmetric PARAFAC:')

def symmetric_decomp(tensor):
    for rank in range(1, 10):
        w, A = tl.decomposition.symmetric_parafac_power_iteration(tensor,
            rank=rank)
        T_hat = np.einsum('xn,yn,zn->xyz', (w[None,:] * A), A, A)
        error = ((tensor - T_hat)**2).sum()
        print(f'decomp: rank {rank}, error {error}')
        if error < 1e-3:
            break
    # Canonicalize the decomposition
    w_sort = np.argsort(w)
    w = w[w_sort]
    A = A[:,w_sort]
    return w, A

print('source:')
ws, As = symmetric_decomp(Ms)
print('target:')
wt, At = symmetric_decomp(Mt)

# We should have that At = T As
T_hat = At @ np.linalg.pinv(As)
print(T_hat)