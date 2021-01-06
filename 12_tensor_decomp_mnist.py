"""
Q: Can we use tensor decomposition to translate MNIST?

A: No. It seems that PARAFAC can't find a good low-rank approximation for the
moment tensors (possibly none exists). In the decomposition we do find, the
eigenvalues don't line up very well, so translation seems hopeless.
"""

import lib
import torch

PCA_DIM = 4

X_source, _, X_target, _ = lib.datasets.colored_mnist()

source_pca = lib.pca.PCA(X_source, PCA_DIM, whiten=True)
target_pca = lib.pca.PCA(X_target, PCA_DIM, whiten=True)
X_source = source_pca.forward(X_source)
X_target = target_pca.forward(X_target)

# # Apply random orthogonal transforms for optimization reasons.
# W1 = lib.ops.random_orthogonal_matrix(X_source.shape[1])
# W2 = lib.ops.random_orthogonal_matrix(X_target.shape[1])
# X_source = X_source @ W1.T
# X_target = X_target @ W2.T

Ms = lib.tensor_decomp.third_moment(X_source)
Mt = lib.tensor_decomp.third_moment(X_target)

ranks = list(range(1,10)) + [2**i for i in range(4, 10)]
ranks = [r for r in ranks if r <= PCA_DIM**2]
for rank in ranks:
    w, A, B, C, err = lib.tensor_decomp.decomp(Ms, rank)
    if err < 1e-2:
        break

ws, As, Bs, Cs, _ = lib.tensor_decomp.decomp(Ms, rank)
wt, At, Bt, Ct, _ = lib.tensor_decomp.decomp(Mt, rank)

if PCA_DIM <= 8:
    lib.utils.print_tensor('ws', ws)
    lib.utils.print_tensor('wt', wt)
    lib.utils.print_tensor('As', As)
    lib.utils.print_tensor('At', At)

# We should have that At = T As
T_hat = At @ torch.pinverse(As)

lib.utils.save_image_grid(
    source_pca.inverse(X_source[:100]), 'source.png')
X_translated = target_pca.inverse(X_source[:100] @ T_hat.T)
lib.utils.save_image_grid(X_translated, 'translated.png')