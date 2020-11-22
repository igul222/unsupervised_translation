import numpy as np
import os
import torch
from torch import nn, optim, autograd
import tqdm
from sklearn.decomposition import PCA
import lib
from scipy.stats import ortho_group
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F

DATA_DIR = os.path.expanduser('~/data/')
CHECKPOINT_PATH = os.path.expanduser('~/jobs/2020_11_19_212913_job_bigEnDe/checkpoint_step50000.pt')
# CHECKPOINT_PATH = os.path.expanduser('~/jobs/2020_11_19_181459_test_job_enDeOrthogonalUnwhitened/checkpoint.pt')
N_COMPONENTS = 300
PREDICTOR_LR = 1e-2
PREDICTION_STEPS = 10001

def load_word_vectors(filename):
    processed_path = os.path.join(DATA_DIR, f'{filename}.processed')
    if os.path.exists(processed_path):
        words, vectors = torch.load(processed_path)
        return words, vectors
    else:
        words = []
        vectors = []
        with open(os.path.join(DATA_DIR, filename), 'r') as f:
            for line_idx, line in tqdm.tqdm(enumerate(f)):
                if line_idx == 0:
                    continue # First line is some kind of header
                parts = line[:-2].split(' ')
                words.append(parts[0])
                vectors.append(torch.tensor([float(x) for x in parts[1:]]))
        vectors = torch.stack(vectors, dim=0)
        torch.save((words, vectors), processed_path)
        return words, vectors

en_words, en_vectors = load_word_vectors('wiki.en.vec')
de_words, de_vectors = load_word_vectors('wiki.de.vec')

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
        Wx = torch.einsum('inx,ixy->iny', x, self.weight)
        return Wx

en_pca, de_pca, translator, losses_argsort = torch.load(CHECKPOINT_PATH)

def load_sentiment_lexicon(lang, wordlist, n):
    # wordlist = [w.lower() for w in wordlist]
    indices = []
    labels = []
    with open(os.path.join(DATA_DIR, f'positive_words_{lang}.txt'), 'r') as f:
        positive_words = [line[:-1] for line in f]
    with open(os.path.join(DATA_DIR, f'negative_words_{lang}.txt'), 'r') as f:
        negative_words = [line[:-1] for line in f]
    assert(len(positive_words) >= n//2)
    assert(len(negative_words) >= n//2)
    for word in positive_words[:n//2]:
        indices.append(wordlist.index(word.lower()))
        labels.append(1)
    for word in negative_words[:n//2]:
        indices.append(wordlist.index(word.lower()))
        labels.append(0)
    rng_state = np.random.get_state()
    np.random.shuffle(indices)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    return np.array(indices), np.array(labels)

en_indices, en_labels = load_sentiment_lexicon('en', en_words, 2000)
de_indices, de_labels = load_sentiment_lexicon('de', de_words, 2000)

import pdb; pdb.set_trace()