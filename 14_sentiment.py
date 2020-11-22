"""
Put the following files into DATA_DIR:
positive_words_en.txt
negative_words_en.txt
positive_words_de.txt
negative_words_de.txt
These files can be downloaded from:
https://sites.google.com/site/datascienceslab/projects/multilingualsentiment
"""

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

en_indices_train, en_indices_test = en_indices[:1000], en_indices[1000:]
en_labels_train, en_labels_test = en_labels[:1000], en_labels[1000:]
de_indices_train, de_indices_test = de_indices[:1000], de_indices[1000:]
de_labels_train, de_labels_test = de_labels[:1000], de_labels[1000:]

en_vectors_train = en_vectors[en_indices_train]
en_vectors_test = en_vectors[en_indices_test]
model = LogisticRegression().fit(en_vectors_train, en_labels_train)
print('en train acc', model.score(en_vectors_train, en_labels_train))
print('en test acc', model.score(en_vectors_test, en_labels_test))

en_vectors_train = en_pca.transform(en_vectors[en_indices_train])
en_vectors_test = en_pca.transform(en_vectors[en_indices_test])
model = LogisticRegression().fit(en_vectors_train, en_labels_train)
print('pca en train acc', model.score(en_vectors_train, en_labels_train))
print('pca en test acc', model.score(en_vectors_test, en_labels_test))

for idx in losses_argsort:
    T = translator.weight.detach().cpu().numpy()[idx]
    translated_en_vectors_train = en_pca.transform(en_vectors[en_indices_train]) @ T
    de_vectors_test = de_pca.transform(de_vectors[de_indices_test])
    model = LogisticRegression().fit(translated_en_vectors_train, en_labels_train)
    print(f'{idx} translated en pca train acc', model.score(translated_en_vectors_train, en_labels_train))
    print(f'{idx} pca de test acc', model.score(de_vectors_test, de_labels_test))

# Worst-case over top 2 translations

translators = [
    translator.weight[losses_argsort[i]]
    for i in range(10)
]
en_labels_train = torch.tensor(en_labels_train).cuda()
en_vectors_train_pca = torch.tensor(en_pca.transform(en_vectors[en_indices_train])).float().cuda()
predictor = nn.Linear(N_COMPONENTS, 1).cuda()
pred_opt = optim.Adam(predictor.parameters(), lr=PREDICTOR_LR, weight_decay=0.001)
def predictor_forward():
    xs_translated = [en_vectors_train_pca @ T for T in translators]
    losses = torch.stack([ F.binary_cross_entropy_with_logits(predictor(xt)[:,0], en_labels_train.float())
        for xt in xs_translated], dim=0)
    return losses.max()

def calculate_acc(vectors, labels):

    return 1 - torch.abs(
        (predictor(vectors)[:,0] > 0).float() - labels
    ).mean()
# de_vectors_test = torch.tensor(torch.tensor(en_pca.transform(en_vectors[en_indices_test])).cuda().float() @ translators[0]).cuda().float()
# de_labels_test = torch.tensor(en_labels_test).cuda().float()
de_vectors_test = torch.tensor(de_pca.transform(de_vectors[de_indices_test])).cuda().float()
de_labels_test = torch.tensor(de_labels_test).cuda().float()


lib.print_row('step', 'loss', 'test acc')
loss_vals = []
for step in range(PREDICTION_STEPS):
    loss = predictor_forward()
    pred_opt.zero_grad()
    loss.backward()
    pred_opt.step()
    loss_vals.append(loss.item())
    if step % 1000 == 0:
        acc = calculate_acc(de_vectors_test, de_labels_test)
        lib.print_row(step, np.mean(loss_vals), acc)
        loss_vals = []