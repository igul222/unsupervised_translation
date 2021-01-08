"""
Translation by tensor decomposition on toy examples with different kinds of
symmetries:
1. No symmetries (unique translation)
2. Reflection along one axis
3. Rotation of a 2D subspace by a fixed angle
4. Rotation of a 2D subspace by any angle.

Result: The fixed-angle-rotation case fails (a single translation is returned),
but the other cases identify the subspace nicely.
"""

import lib
import numpy as np
import os
import sys
import torch

N = 10*1000*1000

def half_parabola(x):
    return torch.clamp(x, min=0).pow(2)

Z1 = torch.randn((N,), device='cuda')
Z2 = torch.randn((N,), device='cuda')

# Toy example 1: no symmetry (unique translation)
X_unique = torch.stack([Z1, half_parabola(Z1)], dim=1)

# Toy example 2: reflectional symmetry
X_reflection = torch.stack([Z1, half_parabola(Z2)], dim=1)

# Toy example 3: fixed-angle rotational symmetry
X_rotation = torch.zeros((N, 3), device='cuda')
X_rotation[:N//3, 0] = float(np.cos(0))
X_rotation[:N//3, 1] = float(np.sin(0))
X_rotation[N//3+1:2*N//3, 0] = float(np.cos(2*np.pi/3))
X_rotation[N//3+1:2*N//3, 1] = float(np.sin(2*np.pi/3))
X_rotation[2*N//3+1:, 0] = float(np.cos(4*np.pi/3))
X_rotation[2*N//3+1:, 1] = float(np.sin(4*np.pi/3))
X_rotation[:, 2] = half_parabola(Z1)
X_rotation += 0.01 * torch.randn(X_rotation.shape, device='cuda')

# Toy example 4: any-angle (entire 2D subspace) rotational symmetry
X_rotation_any = torch.randn((N, 3), device='cuda')
X_rotation_any[:, 2] = half_parabola(X_rotation_any[:, 2])

toy_examples = [
    ('unique',                  X_unique),
    ('reflection',              X_reflection),
    ('rotation_fixed_angle',    X_rotation),
    ('rotation_any_angle',      X_rotation_any)
]

for name, X_source in toy_examples:
    print('-'*80)
    print(f'Toy example: {name}')

    # Whiten X_source
    source_pca = lib.pca.PCA(X_source, 2, whiten=True)
    X_source = source_pca.forward(X_source)

    # Ground-truth orthogonal translation T
    T_groundtruth = lib.ops.random_orthogonal_matrix(2)
    X_target = X_source @ T_groundtruth.T

    # Independent samples in each distribution
    X_source = X_source[::2]
    X_target = X_target[1::2]

    Ms = lib.tensor_decomp.third_moment(X_source)
    Mt = lib.tensor_decomp.third_moment(X_target)

    print('PARAFAC:')

    for rank in range(1, 11):
        _, _, _, _, error = lib.tensor_decomp.decomp(Ms, rank)
        if error < 1e-2:
            break

    ws, As, Bs, Cs, _ = lib.tensor_decomp.decomp(Ms, rank)
    wt, At, Bt, Ct, _ = lib.tensor_decomp.decomp(Mt, rank)

    lib.utils.print_tensor('ws', ws)
    lib.utils.print_tensor('wt', wt)
    lib.utils.print_tensor('As', As)
    lib.utils.print_tensor('At', At)

    # We should have that At = T As
    T_hat = At @ torch.pinverse(As)
    lib.utils.print_tensor('T_groundtruth.T @ T_hat (should be identity)',
        T_groundtruth.T @ T_hat)

    print('Symmetric PARAFAC:')

    for rank in range(1, 11):
        _, _, error = lib.tensor_decomp.symmetric_decomp(Ms, rank)
        if error < 1e-2:
            break

    lib.utils.print_tensor('ws', ws)
    lib.utils.print_tensor('wt', wt)
    lib.utils.print_tensor('As', As)
    lib.utils.print_tensor('At', At)

    ws, As, _ = lib.tensor_decomp.symmetric_decomp(Ms, rank)
    wt, At, _ = lib.tensor_decomp.symmetric_decomp(Mt, rank)

    # We should have that At = T As
    T_hat = At @ torch.pinverse(As)
    lib.utils.print_tensor('T_groundtruth.T @ T_hat (should be identity)',
        T_groundtruth.T @ T_hat)
