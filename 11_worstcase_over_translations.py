"""
'Worst-case Over Translations' algorithm on toy data.
First we run many unsupervised translation attempts, then we learn a predictor
which minimizes the worst-case over the recovered translations.
"""

import numpy as np
import torch
from torch import nn, optim, autograd
import lib
from lib import ops, utils, datasets, wgan_translation

DATASET_SIZE = 100 * 1000
N_TRANSLATIONS = 50
TRANSLATION_STEPS = 3001
N_HID = 128

LOSS_THRESHOLD = 0.1

PREDICTOR_BATCH_SIZE = 64
PREDICTOR_STEPS = 5001
PREDICTOR_LR = 1e-2
EVAL_BATCH_SIZE = 10*1000

def half_parabola(x):
    return np.maximum(0, x)**2

def make_dataset(domain):
    X1 = np.random.randn(DATASET_SIZE, 1)
    X2 = np.random.randn(DATASET_SIZE, 1)
    X3 = half_parabola(X2)
    X4 = np.random.uniform(low=-1, high=1, size=(DATASET_SIZE, 1))
    if domain == 'source':
        X = np.concatenate([X1, X2, X3, X4], axis=1)
    elif domain == 'target':
        X = np.concatenate([X3, X4, X1, X2], axis=1)
    y = (X1 + X2).reshape((-1))
    return torch.tensor(X).float().cuda(), torch.tensor(y).float().cuda()

X_source, y_source = make_dataset('source')
X_target, y_target = make_dataset('target')

# Step 1: Learn many translations.

translations, losses = wgan_translation.translate(X_source, X_target,
    N_TRANSLATIONS, n_steps=TRANSLATION_STEPS, n_hid=N_HID)

# Step 2: Identify a subset of good translations.

translations = [translations[i] for i in range(translations.shape[0])
    if losses[i] <= LOSS_THRESHOLD]

# Step 3: Minimize worst-case risk over them.

predictor = nn.Linear(X_source.shape[1], 1).cuda()
def forward():
    xs, ys = ops.get_batch([X_source, y_source], PREDICTOR_BATCH_SIZE)
    xs_translated = [xs @ trans.detach() for trans in translations]
    losses = torch.stack([(predictor(xt)[:,0] - ys).pow(2).mean()
        for xt in xs_translated], dim=0)
    return losses.max()
utils.train_loop(predictor, forward, PREDICTOR_STEPS, PREDICTOR_LR)

# Step 4: Final evaluation on the target distribution

xt, yt = ops.get_batch([X_target, y_target], EVAL_BATCH_SIZE)
test_loss = (predictor(xt)[:,0] - yt).pow(2).mean()
print('test loss', test_loss)
utils.print_tensor('predictor.weight', predictor.weight)
utils.print_tensor('predictor.bias', predictor.bias)