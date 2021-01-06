"""
'Worst-case Over Translations' algorithm on word-level sentiment classification,
using pretrained Fasttext Wikipedia vectors and a sentiment lexicon.
Source is English, target is German.

Results:

We usually achieve slightly better accuracy than any individual
translation, but it remains to be seen whether we outperform smarter baselines
which make use of multiple translations. For example:
- Average the top-K translation maps to make a new map
- ERM on pooled data from top-K translations
- Ensemble of predictors corresponding to each of top-K translations

Also, energy distance is pretty crap at identifying the good translations.
We should replace it with something better (NND).
"""

import lib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

N_WORDS = 200*1000
PCA_DIM = 256

TRANSLATION_BATCH_SIZE = 1024
TRANSLATION_LAMBDA_GP = 10.
TRANSLATION_LAMBDA_ORTH = 0.1
TRANSLATION_l2reg_D = 1e-4
# "Debug mode"
# N_TRANSLATIONS = 4
# TRANSLATION_STEPS = 101
# TRANSLATION_LR_G = 1e-3
# TRANSLATION_LR_D = 1e-3
# "Fast mode"
# N_TRANSLATIONS = 4
# TRANSLATION_STEPS = 20001
# TRANSLATION_LR_G = 1e-3
# TRANSLATION_LR_D = 1e-3
# "Full mode"
N_TRANSLATIONS = 32
TRANSLATION_STEPS = 100001
TRANSLATION_LR_G = 5e-4
TRANSLATION_LR_D = 1e-3

PREDICTION_STEPS = 5001
PREDICTION_LR = 1e-2
PREDICTION_WEIGHT_DECAY = 1e-3

source_words, X_source = lib.datasets.en_word_vectors()
target_words, X_target = lib.datasets.de_word_vectors()
X_source = X_source[:N_WORDS]
X_target = X_target[:N_WORDS]

source_pca = lib.pca.PCA(X_source, PCA_DIM, whiten=True)
target_pca = lib.pca.PCA(X_target, PCA_DIM, whiten=True)
X_source = source_pca.forward(X_source)
X_target = target_pca.forward(X_target)

# Apply random orthogonal transforms for optimization reasons.
W1 = lib.ops.random_orthogonal_matrix(X_source.shape[1])
W2 = lib.ops.random_orthogonal_matrix(X_target.shape[1])
X_source = X_source @ W1.T
X_target = X_target @ W2.T

# Step 1: Learn many translations.

translations, divergences = lib.adversarial.train_translation(
    X_source, X_target, N_TRANSLATIONS,
    batch_size=TRANSLATION_BATCH_SIZE,
    lambda_gp=TRANSLATION_LAMBDA_GP,
    lambda_orth=TRANSLATION_LAMBDA_ORTH,
    lr_g=TRANSLATION_LR_G,
    lr_d=TRANSLATION_LR_D,
    print_freq=5000,
    steps=TRANSLATION_STEPS,
    l2reg_d=TRANSLATION_l2reg_D,
    algorithm='wgan-gp',
)

word_idx = 81 # source_words[81] is "american"
print(f'Nearest neighbors to "{source_words[word_idx]}":')
for idx in torch.argsort(divergences):
    X_trans = X_source @ translations[idx].T
    target_neighbors = lib.ops.nearest_neighbors(X_target, X_trans[word_idx])
    result_list = ", ".join([target_words[j] for j in target_neighbors[:5]])
    print(f'Instance {idx.item()}:', result_list)

# Step 2: Sort by energy distance.

best_indices = torch.argsort(divergences)
translations = translations[best_indices].detach()

# Step 3: Minimize worst-case risk over them.

X_source, y_source = lib.datasets.en_sentiment_lexicon()
X_target, y_target = lib.datasets.de_sentiment_lexicon()
X_source = source_pca.forward(X_source)
X_target = target_pca.forward(X_target)
X_source = X_source @ W1.T
X_target = X_target @ W2.T

def train_worstcase(Xy_datasets):
    predictor = nn.Linear(X_source.shape[1], 1).cuda()
    def forward():
        losses = [
            F.binary_cross_entropy_with_logits(predictor(X)[:,0], y)
            for X,y in Xy_datasets
        ]
        return torch.stack(losses).max()
    opt = optim.Adam(predictor.parameters(), lr=PREDICTION_LR,
        weight_decay=PREDICTION_WEIGHT_DECAY)
    lib.utils.train_loop(forward, opt, PREDICTION_STEPS, quiet=True)
    return predictor

def acc(predictor, X, y):
    return lib.ops.binary_accuracy(predictor(X)[:,0], y).item()

print('Supervised source baseline:',
    acc(train_worstcase([(X_source, y_source)]), X_source, y_source))

print('Supervised target baseline:',
    acc(train_worstcase([(X_target, y_target)]), X_target, y_target))

for i in range(translations.shape[0]):
    Xy_translated = [(X_source @ translations[i].T, y_source)]
    print(f'{i+1}th best translation only:',
        acc(train_worstcase(Xy_translated), X_target, y_target))

for i in range(1, translations.shape[0]+1):
    Xy_translated = [(X_source @ T.T, y_source) for T in translations[:i]]
    print(f'Worst-case over top {i} translations:',
        acc(train_worstcase(Xy_translated), X_target, y_target))