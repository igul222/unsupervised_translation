"""
IRM on Colored MNIST (the IRM version) with a linear featurizer/classifier
and an orthogonal featurizer. Q: Does the dimensionality of the latent
representation affect anything?

A: Not by much, apparently.
"""

import argparse
import lib
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd

INPUT_DIM = 2*196
PCA_DIM = 64

envs = lib.datasets.irm_colored_mnist()

X_pooled = torch.cat([envs[0]['images'], envs[1]['images']], dim=0)
X_pca = lib.pca.PCA(X_pooled, PCA_DIM, whiten=True)
for env in envs:
    env['images'] = X_pca.forward(env['images'])

for REPR_DIM in [1, 8, 48, 63]:
    for LAMBDA in [1e1, 1e2, 1e4]:

        print('-'*80)
        print(f'REPR_DIM={REPR_DIM}, LAMBDA={LAMBDA}')

        featurizer = nn.Linear(PCA_DIM, REPR_DIM, bias=False).cuda()
        torch.nn.init.orthogonal_(featurizer.weight)
        classifier = nn.Linear(REPR_DIM, 1, bias=False).cuda()

        params = list(featurizer.parameters()) + list(classifier.parameters())
        opt = optim.Adam(params)

        eye = torch.eye(REPR_DIM, device='cuda')

        lib.utils.print_row('step', 'train_nll', 'train_acc', 'train_penalty',
            'test_acc')
        for step in range(5001):
            lib.utils.enforce_orthogonality(featurizer.weight)

            for env in envs:
                feats = featurizer(env['images'])
                ones = torch.ones((1, REPR_DIM), device='cuda')
                ones.requires_grad = True
                logits = classifier(feats * ones)
                env['nll'] = F.binary_cross_entropy_with_logits(logits,
                    env['labels'])
                env['acc'] = lib.ops.binary_accuracy(logits, env['labels'])
                grad = autograd.grad(env['nll'], [ones], create_graph=True)[0]
                env['penalty'] = grad.pow(2).sum()

            train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
            train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
            train_penalty = torch.stack([envs[0]['penalty'],
                envs[1]['penalty']]).mean()

            loss = train_nll.clone()
            loss /= LAMBDA # Keep gradients in a reasonable range
            loss += train_penalty

            opt.zero_grad()
            loss.backward()
            opt.step()

            test_acc = envs[2]['acc']
            if step % 1000 == 0:
                lib.utils.print_row(
                    step,
                    train_nll,
                    train_acc,
                    train_penalty,
                    test_acc
                )