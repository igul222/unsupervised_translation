"""
Is ICA invariant (up to permutation and scaling) to invertible linear
transformations on real-world data?
Answer: Potentially yes! (Up to this test's ability to determine...)

Thoughts: is it weird that the logcosh-values are so uniformly in the
range [0.6, 0.8]?
"""

import numpy as np
from sklearn.decomposition import FastICA
from torchvision import datasets
import lib
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_COMPONENTS = 64

mnist = datasets.MNIST('/tmp', train=True, download=True)
rng_state = np.random.get_state()
np.random.shuffle(mnist.data.numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist.targets.numpy())
X1 = (mnist.data.reshape((60000, 784)).float() / 256.).numpy()

W = np.random.randn(784, 784)
X2 = np.dot(X1, W)

Z3 = np.random.laplace(size=(60000, 64))
W = np.random.randn(64, 784)
X3 = np.dot(Z3, W)

def logcosh(x):
    return torch.log(torch.cosh(x))

def get_logcosh_vals(X):
    ica = FastICA(n_components=N_COMPONENTS)
    ica.fit(X)
    Z = torch.tensor(ica.transform(X))
    # For some reason, sklearn makes Z very tiny, so we scale it up.
    Z *= 100.
    return torch.tensor(
        sorted([logcosh(Z[:,i]).mean().item() for i in range(N_COMPONENTS)])
    )

vals_X1 = get_logcosh_vals(X1)
vals_X2 = get_logcosh_vals(X2)
vals_X3 = get_logcosh_vals(X3)

# If ICA is indeed invariant, then vals_X1 and vals_X2 should be the same, up to
# some tolerance. vals_X3 is a control.

print('W1(X1, X2)', lib.wasserstein_1d(vals_X1, vals_X2))
print('X1.min(), X2.min()', vals_X1.min(), vals_X2.min())
print('X1.max(), X2.max()', vals_X1.max(), vals_X2.max())

print('W1(X1, X3)', lib.wasserstein_1d(vals_X1, vals_X3))
print('X1.min(), X3.min()', vals_X1.min(), vals_X3.min())
print('X1.max(), X3.max()', vals_X1.max(), vals_X3.max())
