"""
Basically just the inner-loop of Algorithm 2.
"""

import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import lib
from lib import ops, utils, datasets, pca, worstcase_translation
import argparse
import geomloss
import time
import sys
import copy
from typing import List
import collections

LATENT_DIM = 64
PCA_DIM = 64
N_INSTANCES = 8

X_source, y_source, X_target, y_target = datasets.colored_mnist()
X_target, y_target, X_test, y_test = datasets.split(X_target, y_target, 0.9)

source_pca = pca.PCA(X_source, PCA_DIM, whiten=True)
target_pca = pca.PCA(X_target, PCA_DIM, whiten=True)
X_source = source_pca.forward(X_source)
X_target = target_pca.forward(X_target)
X_test = target_pca.forward(X_test)

# Apply random orthogonal transforms for optimization reasons.
W1 = ops.random_orthogonal_matrix(PCA_DIM)
W2 = ops.random_orthogonal_matrix(PCA_DIM)
X_source = X_source @ W1.T
X_target = X_target @ W2.T
X_test   = X_test   @ W2.T

source_rep = nn.Linear(PCA_DIM, LATENT_DIM, bias=False).cuda()
torch.nn.init.orthogonal_(source_rep.weight)
classifier = nn.Linear(LATENT_DIM, 10, bias=False).cuda()

# Random target rep. Not ideal, but whatever.
target_rep = nn.Linear(PCA_DIM, LATENT_DIM, bias=False).cuda()
torch.nn.init.orthogonal_(target_rep.weight)

# Train the classifier
def forward():
    return F.cross_entropy(classifier(source_rep(X_source)), y_source)
utils.train_loop(classifier, forward, 1001, 1e-2, quiet=False)

# Find the worst-case-translations
alt_target_rep, energy_dists, tvs = worstcase_translation.WorstCaseTranslation(
    n_instances=N_INSTANCES,
    input_dim=PCA_DIM,
    latent_dim=LATENT_DIM
).train(
    X_source, X_target, source_rep, target_rep, classifier, verbose=True
)

# Calculate test accuracies for each
with torch.no_grad():
    logits = classifier(alt_target_rep(X_test.expand(N_INSTANCES, -1, -1)))
    test_accs = ops.multiclass_accuracy(logits,
        y_test.expand(N_INSTANCES, -1)).mean(dim=1)

utils.print_row('instance', 'energy dist', 'tv', 'test acc')
for idx in torch.argsort(energy_dists):
    utils.print_row(idx, energy_dists[idx], tvs[idx], test_accs[idx])