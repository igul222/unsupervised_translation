"""
'Worst-case Over Translations' algorithm on toy data: first we run many
unsupervised translation attempts, then we learn a predictor which minimizes
the worst-case over the recovered translations.

This toy example has a reflectional symmetry, so there are two possible
translations. The target depends on two variables, one of which is translated
unambiguously and the other of which isn't. The optimal thing to do is to throw
out the ambiguous variable and predict only based on the one with an unambiguous
translation.

Result: it works!
"""
import lib
import numpy as np
import torch
from torch import nn, optim

DATASET_SIZE = 10*1000
N_TRANSLATIONS = 64
TRANSLATION_STEPS = 1001
TRANSLATION_LAMBDA_GP = 0.1
TRANSLATION_LR_G = 1e-2

PREDICTION_BATCH_SIZE = 1024
PREDICTION_STEPS = 1001
PREDICTION_LR = 1e-2

def half_parabola(x):
    return torch.clamp(x, min=0).pow(2)

def make_dataset(domain):
    X1 = torch.randn((DATASET_SIZE, 1), device='cuda')
    X2 = torch.randn((DATASET_SIZE, 1), device='cuda')
    X3 = half_parabola(X2)
    X4 = (2*torch.rand((DATASET_SIZE, 1), device='cuda')) - 1
    if domain == 'source':
        X = torch.cat([X1, X2, X3, X4], dim=1)
    elif domain == 'target':
        X = torch.cat([X3, X4, X1, X2], dim=1)
    y = (X1 + X2).view(-1)
    return X, y

X_source, y_source = make_dataset('source')
X_target, y_target = make_dataset('target')

# Step 1: Learn many translations.

translations, divergences = lib.adversarial.train_translation(
    X_source, X_target, N_TRANSLATIONS,
    lambda_gp=TRANSLATION_LAMBDA_GP,
    lr_g=TRANSLATION_LR_G,
    steps=TRANSLATION_STEPS)

# Step 2: Sort by energy distance.

best_indices = torch.argsort(divergences)
translations = translations[best_indices].detach()

# Step 3: Minimize worst-case risk over them.

predictor = nn.Linear(X_source.shape[1], 1).cuda()
def forward():
    Xs, ys = lib.ops.get_batch([X_source, y_source], PREDICTION_BATCH_SIZE)
    Xs = Xs[None,:,:].expand(translations.shape[0], -1, -1)
    ys = ys[None,:].expand(translations.shape[0], -1)
    Xs_translated = torch.bmm(Xs, translations.permute(0,2,1))
    losses = (predictor(Xs_translated)[:,:,0] - ys).pow(2).mean(dim=1)
    return losses.max()
opt = optim.Adam(predictor.parameters(), lr=PREDICTION_LR)
lib.utils.train_loop(forward, opt, PREDICTION_STEPS, lr=PREDICTION_LR)

# Step 4: Final evaluation on the target distribution

test_loss = (predictor(X_target)[:,0] - y_target).pow(2).mean()
print('test loss', test_loss)
lib.utils.print_tensor('predictor.weight', predictor.weight)
lib.utils.print_tensor('predictor.bias', predictor.bias)