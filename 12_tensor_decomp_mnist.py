"""
Can we use tensor decomposition to translate MNIST?
Answer: seemingly not.
"""

import numpy as np
from sklearn.decomposition import PCA
import tensorly as tl
import tensorly.decomposition
import lib
import os
import sys
from torchvision import datasets
import torch

OUTPUT_DIR = 'outputs/12_tensor_decomp_mnist'
PCA_DIMS = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.stdout = lib.Tee(f'{OUTPUT_DIR}/output.txt')

np.set_printoptions(suppress=True, precision=4, linewidth=200)

tl.set_backend('pytorch')

mnist = datasets.MNIST('/tmp', train=True, download=True)
rng_state = np.random.get_state()
np.random.shuffle(mnist.data.numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist.targets.numpy())
mnist_data = (mnist.data.reshape((60000, 784)).float() / 256.).cuda()

mnist_source = mnist_data[::2]
groundtruth_translation = torch.randn((784, 784)).cuda()
# groundtruth_translation = torch.eye(784).cuda()
mnist_target = (mnist_data[1::2] @ groundtruth_translation.T)

source_pca = PCA(n_components=PCA_DIMS, whiten=True).fit(
    mnist_source.cpu().numpy())
mnist_source_whitened = torch.tensor(
    source_pca.transform(mnist_source.cpu().numpy())).cuda()
target_pca = PCA(n_components=PCA_DIMS, whiten=True).fit(
    mnist_target.cpu().numpy())
mnist_target_whitened = torch.tensor(
    target_pca.transform(mnist_target.cpu().numpy())).cuda()

def third_moment(X):
    M = torch.einsum('nx,ny,nz->xyz', X, X, X) / X.shape[0]
    return M

Ms = third_moment(mnist_source_whitened)
Mt = third_moment(mnist_target_whitened)

def decomp(tensor, rank):
    (w, (A, B, C)), errors = tl.decomposition.parafac(tensor, rank=rank,
        return_errors=True, normalize_factors=True)
    print(f'decomp: rank {rank}, error {errors[-1]}')
    # Canonicalize the decomposition
    w_sort = torch.argsort(w)
    w = w[w_sort]
    A = A[:,w_sort]
    B = B[:,w_sort]
    C = C[:,w_sort]
    return w, A, B, C, errors[-1]

def unknown_rank_decomp(tensor):
    for rank in list(range(1,10)) + [2**i for i in range(4, 10)]:
        w, A, B, C, err = decomp(tensor, rank)
        if err < 1e-2:
            break
    return w, A, B, C, rank

print('source:')
ws, As, Bs, Cs, rank = unknown_rank_decomp(Ms)
print('target:')
wt, At, Bt, Ct, _ = decomp(Mt, rank)

lib.print_tensor('ws', ws)
lib.print_tensor('wt', wt)
lib.print_tensor('As', As)
lib.print_tensor('At', At)

# We should have that At = T As
T_hat = At.cpu().numpy() @ np.linalg.pinv(As.cpu().numpy())
lib.print_tensor('T_hat', T_hat)

lib.save_image_grid_mnist(mnist_source[:100].cpu().numpy(),
    f'{OUTPUT_DIR}/source.png')
translated = target_pca.inverse_transform(
    (mnist_source_whitened[:100].cpu().numpy() @ T_hat.T))
# For visualization, we map translated examples back into the source domain
# using the ground-truth translation map.
inverse_groundtruth_map = torch.inverse(groundtruth_translation).T.cpu().numpy()
translated = translated @ inverse_groundtruth_map
lib.save_image_grid_mnist(translated, f'{OUTPUT_DIR}/translated.png')