"""
Before running, set DATA_DIR to a directory containing these files:
https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.vec

unwhitened orthogonal: ~/jobs/2020_11_27_121848_word_embeddings_translation_unwhitened_orthogonal/result.pt
whitened orthogonal: ~/jobs/2020_11_27_153800_word_embeddings_translation_whitened_orthogonal/result.pt
"""

import numpy as np
import os
import torch
from torch import nn, optim, autograd
import tqdm
from sklearn.decomposition import PCA
from lib import utils, ops, datasets, wgan_translation, pca
from scipy.stats import ortho_group

WHITEN = True
ORTH_PENALTY = 0.01 # 0 for arbitrary map, 0.01 for orthogonal map
EN_TO_ITSELF = False # en->en if True, en->de otherwise

N_WORDS = 200000
N_PCA = 300
N_TRANSLATIONS = 100
STEPS = 50001

source_words, source_vectors = datasets.en_word_vectors()
if EN_TO_ITSELF:
    print('WARNING: Translating EN to itself!')
    target_words, target_vectors = source_words, source_vectors
else:
    target_words, target_vectors = datasets.de_word_vectors()

source_vectors = source_vectors[:N_WORDS]
target_vectors = target_vectors[:N_WORDS]

source_pca = pca.PCA(source_vectors, N_PCA, whiten=WHITEN)
target_pca = pca.PCA(target_vectors, N_PCA, whiten=WHITEN)
source_vectors = source_pca.forward(source_vectors)
target_vectors = target_pca.forward(target_vectors)

translations, losses = wgan_translation.translate(
    source_vectors, target_vectors,
    N_TRANSLATIONS, n_steps=STEPS, orth_penalty=ORTH_PENALTY
)

# For each map, print the target-language nearest neighbors to 'american'

losses_argsort = torch.argsort(losses)
print('Losses:', losses[losses_argsort])
print('Nearest neighbors to "american":')
for idx in losses_argsort:
    translated = source_vectors @ translations[idx]
    word_idx, source_word = (81, source_words[81]) # "american"
    translated_vec = translated[word_idx]
    target_neighbors = ops.nearest_neighbors(target_vectors, translated_vec)[:3]
    print(idx, [target_words[j] for j in target_neighbors])

torch.save(
    (source_pca, target_pca, translations, losses),
    'result.pt'
)