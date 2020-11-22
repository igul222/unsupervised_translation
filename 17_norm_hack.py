"""
Aligns MNIST with itself (i.e. the correct solution is the identity matrix).
Method: whiten, then square the norm of each point, then align the PCs.
I didn't bother to correct flipped PCs, so some entries of the recovered
alignment are -1 rather than 1.
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
OUTPUT_DIR = 'outputs/17_norm_hack'

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.stdout = lib.Tee(f'{OUTPUT_DIR}/output.txt')

torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)

with torch.no_grad():

    mnist = datasets.MNIST('/tmp', train=True, download=True)
    rng_state = np.random.get_state()
    np.random.shuffle(mnist.data.numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist.targets.numpy())
    mnist_data = (mnist.data.reshape((60000, 784)).float()/256.).clone().cuda()

    source = mnist_data[::2]
    target = mnist_data[1::2]

    def whiten(X):
        X = X.detach().cpu().numpy()
        pca = PCA(n_components=PCA_DIMS, whiten=True)
        return torch.tensor(pca.fit_transform(X)).float().cuda(), pca

    source, source_pca = whiten(source)
    target, target_pca = whiten(target)
 
    def square_norms(X):
        return X * X.norm(p=2, dim=1, keepdim=True)

    As = torch.tensor(whiten(square_norms(source))[1].components_.T).cuda()
    At = torch.tensor(whiten(square_norms(target))[1].components_.T).cuda()

    # we should have At = T As
    print('Recovered translation (should be identity):')
    T_hat = At @ torch.pinverse(As)
    print(T_hat)
    err = ((T_hat @ As) - At).pow(2).sum()
    print('error:', err.item())