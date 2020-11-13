"""
'Worst-case Over Translations' algorithm on toy data.
First we run many unsupervised translation attempts, then we learn a predictor
which minimizes the worst-case over the recovered translations.

State of this code: messy but working.
"""

import numpy as np
import torch
from torch import nn, optim, autograd
from torchvision import datasets
import torch.nn.functional as F
import lib
import os
import sys

DATASET_SIZE = 100 * 1000
N_INSTANCES = 100
X_DIM = 4
HIDDEN_DIM = 64
BATCH_SIZE = 64
WGANGP_LAMDA = 0.1
LR = 1e-3
EVAL_BS = 1000
OUTPUT_DIR = 'outputs/11_wot'

TRANSLATION_STEPS = 3001
TRANSLATION_EVAL_STEPS = 2001

DIFFERENCE_THRESHOLD = 0.1
LOSS_THRESHOLD = -0.1

PREDICTION_STEPS = 10001
PREDICTOR_LR = 1e-2

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.stdout = lib.Tee(f'{OUTPUT_DIR}/output.txt')

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

class MultipleLinear(nn.Module):
    def __init__(self, dim_in, dim_out, n_instances, bias=True):
        super().__init__()
        if bias:
            self.weight = nn.Parameter(torch.randn((n_instances, dim_in+1,
                dim_out)) / float(np.sqrt(dim_in)) )
        else:
            self.weight = nn.Parameter(torch.randn((n_instances, dim_in,
                dim_out)) / float(np.sqrt(dim_in)) )
        self.bias = bias
    def forward(self, x):
        if self.bias:
            bias_var = torch.ones_like(x[:, :, 0:1]).cuda()
            x = torch.cat([x, bias_var], dim=2)
        Wx = torch.einsum('ixy,inx->iny', self.weight, x)
        return Wx

# Step 1: Train many translation models simultaneously

def init_translation_model():
    translator = MultipleLinear(X_DIM, X_DIM, N_INSTANCES, bias=False).cuda()
    discriminator = nn.Sequential(
        MultipleLinear(X_DIM, HIDDEN_DIM, N_INSTANCES),
        nn.ReLU(),
        MultipleLinear(HIDDEN_DIM, HIDDEN_DIM, N_INSTANCES),
        nn.ReLU(),
        MultipleLinear(HIDDEN_DIM, 1, N_INSTANCES)
    ).cuda()

    translator_opt = optim.Adam(translator.parameters(), lr=LR,
        betas=(0., 0.99))
    disc_opt = optim.Adam(discriminator.parameters(), lr=LR,
        betas=(0., 0.99))
    return translator, discriminator, translator_opt, disc_opt

def translation_forward(bs=BATCH_SIZE):
    x_source = lib.get_batch(X_source, N_INSTANCES * bs).view(
        N_INSTANCES, bs, X_DIM)
    x_target = lib.get_batch(X_target, N_INSTANCES * bs).view(
        N_INSTANCES, bs, X_DIM)
    x_translated = translator(x_target)

    disc_real = discriminator(x_source)
    disc_fake = discriminator(x_translated)
    loss = disc_real.mean(dim=[1,2]) - disc_fake.mean(dim=[1,2])
    epsilon = torch.rand(N_INSTANCES, bs, 1).cuda()
    interps = (epsilon*x_source) + ((1-epsilon)*x_translated)
    disc_interps = discriminator(interps)
    grad = autograd.grad(disc_interps.sum(), [interps], create_graph=True)[0]
    grad_norm = (grad.pow(2).sum(dim=2) + 1e-6).sqrt()
    gp = (grad_norm - 1).pow(2).mean(dim=1)
    return (loss + (WGANGP_LAMDA * gp))

translator, discriminator, translator_opt, disc_opt = init_translation_model()
lib.print_row('step', 'loss')
loss_vals = []
for step in range(TRANSLATION_STEPS):
    for inner_step in range(5):
        loss = translation_forward().mean()
        translator_opt.zero_grad()
        disc_opt.zero_grad()
        loss.backward()
        disc_opt.step()
    loss = translation_forward().mean()
    translator_opt.zero_grad()
    disc_opt.zero_grad()
    (-loss).backward()
    translator_opt.step()
    loss_vals.append(loss.item())
    if step % 1000 == 0:
        lib.print_row(step, np.mean(loss_vals))
        loss_vals = []

# Retrain the discriminator from scratch to estimate a divergence

_, discriminator, _, disc_opt = init_translation_model()
lib.print_row('step', 'loss')
loss_vals = []
for step in range(TRANSLATION_EVAL_STEPS):
    loss = translation_forward().mean()
    disc_opt.zero_grad()
    loss.backward()
    disc_opt.step()
    loss_vals.append(loss.item())
    if step % 1000 == 0:
        lib.print_row(step, np.mean(loss_vals))
        loss_vals = []

# Step 2: Identify a subset of substantially different good translations.

x_target = torch.stack(N_INSTANCES*[lib.get_batch(X_target, EVAL_BS)], dim=0)
x_translated = translator(x_target)
losses = translation_forward(bs=EVAL_BS)
losses_argsort = torch.flip(torch.argsort(losses), [0])
chosen_translations = []
for idx in losses_argsort:
    loss = losses[idx]
    min_difference = torch.tensor(np.inf).float()
    for other_idx in chosen_translations:
        difference = (x_translated[idx] - x_translated[other_idx]).pow(2).mean()
        if difference < min_difference:
            min_difference = difference
    if min_difference > DIFFERENCE_THRESHOLD and loss > LOSS_THRESHOLD:
        chosen_translations.append(idx)

print('Chosen translations:', chosen_translations)

# Step 3: Minimize worst-case risk over them

inv_translations = [torch.inverse(translator.weight[i])
    for i in chosen_translations]

predictor = nn.Linear(X_DIM, 1).cuda()
pred_opt = optim.Adam(predictor.parameters(), lr=PREDICTOR_LR)
def predictor_forward():
    xs, ys = lib.get_batch([X_source, y_source], BATCH_SIZE)
    xs_translated = [xs @ inv_trans.detach() for inv_trans in inv_translations]
    losses = torch.stack([(predictor(xt)[:,0] - ys).pow(2).mean()
        for xt in xs_translated], dim=0)
    return losses.max()

lib.print_row('step', 'loss')
loss_vals = []
for step in range(PREDICTION_STEPS):
    loss = predictor_forward()
    pred_opt.zero_grad()
    loss.backward()
    pred_opt.step()
    loss_vals.append(loss.item())
    if step % 1000 == 0:
        lib.print_row(step, np.mean(loss_vals))
        loss_vals = []

# Step 4: Final evaluation on the target distribution

with torch.no_grad():
    losses = []
    for _ in range(1000):
        xt, yt = lib.get_batch([X_target, y_target], BATCH_SIZE)
        test_loss = (predictor(xt)[:,0] - yt).pow(2).mean()
        losses.append(test_loss.item())
    print('test loss', np.mean(losses))

print('predictor.weight', predictor.weight)
print('predictor.bias', predictor.bias)