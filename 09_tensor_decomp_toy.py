"""
Translation by tensor decomposition on a 2D toy example.
"""

import numpy as np
import sklearn.decomposition
import tensorly as tl
import tensorly.decomposition

N = 10*1000

Y_source = np.random.randn(N)
X_source = np.stack([Y_source, np.exp(Y_source)], axis=0)

# Whiten X_source
X_source -= np.mean(X_source, axis=0)
X_source = sklearn.decomposition.PCA(whiten=True).fit_transform(X_source)

# Ground-truth orthogonal translation T
T_groundtruth = np.array([[0., 1.], [1., 0.]])
X_target = X_source @ T_groundtruth.T

def third_moment(X):
    return np.einsum('nx,ny,nz->xyz', X, X, X) / X.shape[0]

Ms = third_moment(X_source)
Mt = third_moment(X_target)

def decomp(tensor):
    for rank in range(1, 10):
        (w, (A, B, C)), errors = tl.decomposition.parafac(tensor, rank=rank,
            return_errors=True)
        print(f'decomp: rank {rank}, error {errors[-1]}')
        if errors[-1] < 1e-6:
            break
    return w, A, B, C

print('source:')
ws, As, Bs, Cs = decomp(Ms)
print('target:')
wt, At, Bt, Ct = decomp(Mt)

# We should have that At = T As
T_hat_1 = At @ np.linalg.pinv(As)
print(T_hat_1)
# import pdb; pdb.set_trace()
