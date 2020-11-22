"""
Before running, set DATA_DIR to a directory containing these files:
https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.vec

Results so far:
EN to itself, unwhitened, with an orthogonal map: works
EN to itself, whitened, with an orthogonal map: works
EN to itself, unwhitened, with an arbitrary map: probably doesn't work
EN to itself, whitened, with an arbitrary map: probably doesn't work

EN to DE, orthogonal unwhitened: works! (40K steps; 50K+ might improve results further...)
EN to DE, orthogonal whitened: again, signs of life, but even weaker.

checkpoints:
2020_11_19_181459_test_job_enDeOrthogonalUnwhitened/checkpoint.pt (40K steps, seems decent)
test_job_enDeOrthogonalWhitened40K/checkpoint.pt (40K steps)
"""

import numpy as np
import os
import torch
from torch import nn, optim, autograd
import tqdm
from sklearn.decomposition import PCA
import lib
from scipy.stats import ortho_group

DATA_DIR = os.path.expanduser('~/data/')
N_COMPONENTS = 300
DIM = 512
N_VECTORS = 200000
LR = 5e-4
BATCH_SIZE = 256
WGANGP_LAMDA = 1.
N_INSTANCES = 100
ORTH_PENALTY = 0.01 # 0 or 0.01 are good choices
DISC_WEIGHT_DECAY = 1e-4 / N_INSTANCES
STEPS = 50001
WHITEN = False

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
# print('WARNING ALIGNING EN TO ITSELF')

def whiten(X):
    pca = PCA(n_components=N_COMPONENTS, whiten=WHITEN)
    return pca, torch.tensor(pca.fit_transform(X)).float().cuda()

en_pca, en_vectors = whiten(en_vectors[:N_VECTORS])
de_pca, de_vectors = whiten(de_vectors[:N_VECTORS])

def nearest_neighbors(M, v):
    """Returns indices of rows of M which are nearest (by cosine dist) to v."""
    M_normalized = M / (1e-6 + M.norm(p=2, dim=1, keepdim=True))
    dists = M_normalized @ v
    return torch.flip(torch.argsort(dists), [0])

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

# Step 1: Train many translation maps simultaneously

def init_translation_model():
    translator = MultipleLinear(N_COMPONENTS, N_COMPONENTS, N_INSTANCES,
        bias=False).cuda()
    discriminator = nn.Sequential(
        MultipleLinear(N_COMPONENTS, DIM, N_INSTANCES),
        nn.ReLU(),
        MultipleLinear(DIM, DIM, N_INSTANCES),
        nn.ReLU(),
        MultipleLinear(DIM, 1, N_INSTANCES)
    ).cuda()

    translator_opt = optim.Adam(translator.parameters(), lr=LR,
        betas=(0., 0.99))
    disc_opt = optim.Adam(discriminator.parameters(), lr=LR,
        betas=(0., 0.99), weight_decay=DISC_WEIGHT_DECAY)
    return translator, discriminator, translator_opt, disc_opt

def forward(bs=BATCH_SIZE):
    x_source = lib.get_batch(en_vectors, N_INSTANCES * bs).view(
        N_INSTANCES, bs, N_COMPONENTS)
    x_target = lib.get_batch(de_vectors, N_INSTANCES * bs).view(
        N_INSTANCES, bs, N_COMPONENTS)
    x_translated = translator(x_source)

    disc_real = discriminator(x_target)
    disc_fake = discriminator(x_translated)
    loss = disc_real.mean(dim=[1,2]) - disc_fake.mean(dim=[1,2])
    epsilon = torch.rand(N_INSTANCES, bs, 1).cuda()
    interps = (epsilon*x_target) + ((1-epsilon)*x_translated)
    disc_interps = discriminator(interps)
    grad = autograd.grad(disc_interps.sum(), [interps], create_graph=True)[0]
    grad_norm = (grad.pow(2).sum(dim=2) + 1e-6).sqrt()
    gp = (grad_norm - 1).pow(2).mean(dim=1)

    return (loss + (WGANGP_LAMDA * gp))

def translator_orth_penalty():
    _eye_n = torch.eye(N_COMPONENTS, device='cuda')[None,:,:]
    WWT = torch.einsum('iab,ibc->iac',
        translator.weight,
        translator.weight.permute(0,2,1))
    return (WWT - _eye_n).pow(2).sum(dim=[1,2])

translator, discriminator, translator_opt, disc_opt = init_translation_model()
lib.print_row('step', 'loss')
loss_vals = []
for step in range(STEPS):
    for inner_step in range(5):
        translator_opt.zero_grad()
        disc_opt.zero_grad()
        loss = forward().mean()
        loss.backward()
        disc_opt.step()
    translator_opt.zero_grad()
    disc_opt.zero_grad()
    losses = forward()
    trans_loss = ((ORTH_PENALTY * translator_orth_penalty()) + (-losses)).mean()
    trans_loss.backward()
    translator_opt.step()
    loss_vals.append(losses.mean().item())
    if step % 100 == 0:
        lib.print_row(step, np.mean(loss_vals))
        loss_vals = []

        losses = forward(1024)
        losses_argsort = torch.flip(torch.argsort(losses), [0])
        torch.save((en_pca, de_pca, translator, losses_argsort), f'checkpoint_step{step}.pt')

# For the best maps, print the target-language nearest neighbors to 'american'

losses = forward(1024)
losses_argsort = torch.flip(torch.argsort(losses), [0])
print('losses: ', losses[losses_argsort])
for idx in losses_argsort[:10]:
    translated = en_vectors @ translator.weight[idx]
    idx, en_word = (81, en_words[81]) # "american"
    translated_vec = translated[idx]
    de_neighbors = nearest_neighbors(de_vectors, translated_vec)[:3]
    print(en_word, [de_words[j] for j in de_neighbors])