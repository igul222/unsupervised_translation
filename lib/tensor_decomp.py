"""
Helper functions for tensor decomposition experiments.
"""

import numpy as np
import torch
import tensorly as tl
import tensorly.decomposition

def third_moment(X):
    return torch.einsum('nx,ny,nz->xyz', X, X, X) / X.shape[0]

def decomp(tensor, rank):
    tl.set_backend('pytorch')
    (w, (A, B, C)), errors = tl.decomposition.parafac(tensor, rank=rank,
        return_errors=True, normalize_factors=True)
    w_sort = torch.argsort(w)
    w = w[w_sort]
    A = A[:,w_sort]
    B = B[:,w_sort]
    C = C[:,w_sort]
    print(f'decomp: rank {rank}, error {errors[-1]}')
    return w, A, B, C, errors[-1]

def symmetric_decomp(tensor, rank):
    tl.set_backend('pytorch')
    w, A = tl.decomposition.symmetric_parafac_power_iteration(
        tensor, rank=rank)
    w_sort = torch.argsort(w)
    w = w[w_sort]
    A = A[:,w_sort]
    T_hat = torch.einsum('xn,yn,zn->xyz', (w[None,:] * A), A, A)
    error = ((tensor - T_hat)**2).sum()
    return w, A, error
