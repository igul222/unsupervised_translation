"""
Q: Can we use tensor decomposition to translate MNIST?

A: No. It seems that PARAFAC can't find a good low-rank approximation for the
moment tensors (possibly none exists). In the decomposition we do find, the
eigenvalues don't line up very well, so translation seems hopeless.
"""

import torch
from lib import ops, utils, datasets, tensor_decomp, pca

PCA_DIM = 4

X_source, _, X_target, _ = datasets.colored_mnist()

source_pca = pca.PCA(X_source, PCA_DIM, whiten=True)
target_pca = pca.PCA(X_target, PCA_DIM, whiten=True)
X_source = source_pca.forward(X_source)
X_target = target_pca.forward(X_target)

# # Apply random orthogonal transforms for optimization reasons.
# W1 = ops.random_orthogonal_matrix(X_source.shape[1])
# W2 = ops.random_orthogonal_matrix(X_target.shape[1])
# X_source = X_source @ W1.T
# X_target = X_target @ W2.T

Ms = tensor_decomp.third_moment(X_source)
Mt = tensor_decomp.third_moment(X_target)

ranks = list(range(1,10)) + [2**i for i in range(4, 10)]
ranks = [r for r in ranks if r <= PCA_DIM**2]
for rank in ranks:
    w, A, B, C, err = tensor_decomp.decomp(Ms, rank)
    if err < 1e-2:
        break

ws, As, Bs, Cs, _ = tensor_decomp.decomp(Ms, rank)
wt, At, Bt, Ct, _ = tensor_decomp.decomp(Mt, rank)

if PCA_DIM <= 8:
    utils.print_tensor('ws', ws)
    utils.print_tensor('wt', wt)
    utils.print_tensor('As', As)
    utils.print_tensor('At', At)

# We should have that At = T As
T_hat = At @ torch.pinverse(As)

utils.save_image_grid_colored_mnist(
    source_pca.inverse(X_source[:100]), 'source.png')
X_translated = target_pca.inverse(X_source[:100] @ T_hat.T)
utils.save_image_grid_colored_mnist(X_translated, 'translated.png')