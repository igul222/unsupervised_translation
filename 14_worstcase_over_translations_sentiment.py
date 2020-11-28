"""
Put the following files into DATA_DIR:
positive_words_en.txt
negative_words_en.txt
positive_words_de.txt
negative_words_de.txt
These files can be downloaded from:
https://sites.google.com/site/datascienceslab/projects/multilingualsentiment
"""

import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
from lib import utils, ops, datasets, pca
import os

N_PCA = 300
CHECKPOINT_PATH = os.path.expanduser('~/jobs/'
    '2020_11_27_121848_word_embeddings_translation_unwhitened_orthogonal/'
    'result.pt')
L2_PENALTY = 1e-3
LR = 1e-2
STEPS = 5001

WOT_N_TRANSLATIONS = 10 # Pick the top N translations

source_words, source_vectors = datasets.en_word_vectors()
target_words, target_vectors = datasets.de_word_vectors()

source_pca, target_pca, translations, losses = torch.load(CHECKPOINT_PATH)

Xs, ys = datasets.en_sentiment_lexicon()
Xt, yt = datasets.de_sentiment_lexicon()

Xs_train, ys_train, Xs_test, ys_test = datasets.split(Xs, ys, 0.5)
Xt_train, yt_train, Xt_test, yt_test = datasets.split(Xt, yt, 0.5)

def train_and_eval_model(Xy_pairs, X_test, y_test):
    model = nn.Linear(N_PCA, 1).cuda()
    def forward():
        losses = [F.binary_cross_entropy_with_logits(model(X)[:,0], y)
            for X, y in Xy_pairs]
        loss = torch.stack(losses).max()
        return loss + (L2_PENALTY * model.weight.pow(2).sum())
    utils.train_loop(model, forward, STEPS, LR, quiet=True)
    return ops.binary_accuracy(model(X_test)[:,0], y_test).item()

# Baseline: Supervised target-domain

print('Supervised (raw)', train_and_eval_model(
    [(Xt_train, yt_train)],
    Xt_test, yt_test
))

print('Supervised (PCA)', train_and_eval_model(
    [(target_pca.forward(Xt_train), yt_train)],
    target_pca.forward(Xt_test), yt_test
))

# Worst-case over translations

losses_argsort = torch.argsort(losses)
good_translations = translations[losses_argsort[:WOT_N_TRANSLATIONS]]
Xs_translations = [
    (source_pca.forward(Xs_train) @ T).detach()
    for T in good_translations
]

print('Worst-case over translations', train_and_eval_model(
    [(X, ys_train) for X in Xs_translations],
    target_pca.forward(Xt_test), yt_test
))

# Baseline: individual translations

for i, X in enumerate(Xs_translations):
    print(f'{i}th-best translation', train_and_eval_model(
        [(X, ys_train)],
        target_pca.forward(Xt_test), yt_test
    ))