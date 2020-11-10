"""
Translation by tensor decomposition on a toy example with different kinds of
symmetry: reflection along one axis, rotation of a 2D subspace by a fixed angle,
rotation of a 2D subspace by any angle.

The fixed-angle-rotation case fails badly, but the other cases identify the
subspace nicely.
"""

import numpy as np
import sklearn.decomposition
import tensorly as tl
import tensorly.decomposition
import lib
import os
import sys

N = 10*1000*1000
OUTPUT_DIR = 'outputs/06_mnist_translation_identity'

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.stdout = lib.Tee(f'{OUTPUT_DIR}/output.txt')

np.set_printoptions(suppress=True, precision=8)

def half_parabola(x):
    return np.maximum(0, x)**2

Z1 = np.random.randn(N)
Z2 = np.random.randn(N)
X_reflection = np.stack([Z1, half_parabola(Z2)], axis=1)

X_rotation = np.zeros((N, 3))
X_rotation[:N//3, 0] = np.cos(0)
X_rotation[:N//3, 1] = np.sin(1)
X_rotation[N//3+1:2*N//3, 0] = np.cos(2*np.pi/3)
X_rotation[N//3+1:2*N//3, 1] = np.sin(2*np.pi/3)
X_rotation[2*N//3+1:, 0] = np.cos(4*np.pi/3)
X_rotation[2*N//3+1:, 1] = np.sin(4*np.pi/3)
X_rotation[:, 2] = half_parabola(Z1)
X_rotation += 0.01 * np.random.randn(*X_rotation.shape)

X_rotation_any = np.random.randn(N, 3)
X_rotation_any[:, 2] = half_parabola(X_rotation_any[:, 2])

test_cases = [
    ('reflection', X_reflection),
    ('rotation_fixed_angle', X_rotation),
    ('rotation_any_angle', X_rotation_any)
]

for name, X_source in test_cases:
    print('test case:', name)

    # Whiten X_source
    X_source -= np.mean(X_source, axis=0)
    X_source = sklearn.decomposition.PCA(whiten=True).fit_transform(X_source)

    # Ground-truth orthogonal translation T
    if X_source.shape[1] == 2:
        T_groundtruth = np.array([[0., 1.], [1., 0.]])
    elif X_source.shape[1] == 3:
        T_groundtruth = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
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
            T_hat = np.einsum('xn,yn,zn->xyz', (w[None,:] * A), A, A)
            print(f'decomp: rank {rank}, error {errors[-1]}')
            if errors[-1] < 1e-3:
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

    if As.shape == At.shape:
        # We should have that At = T As
        T_hat = At @ np.linalg.pinv(As)
        print(T_hat)
    else:
        print('Not attempting to find a translation; ranks didn\'t match')