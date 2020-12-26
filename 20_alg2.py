"""
"Algorithm 2" on Colored MNIST:
Player 1 chooses an invariance (left and right halves) and a classifier.
Player 2 chooses a different right-half.
Player 3 is a discriminator.

Player 2 tries to maximize a divergence between classifier outputs given its
right-half and Player 1's.
Player 1 tries to minimize classification error and Player 2's objective.

This doesn't seem to work well.
"""

import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import lib
from lib import ops, utils, datasets
import argparse

N_INSTANCES = 64

parser = argparse.ArgumentParser()
parser.add_argument('--randomize_hparams', action='store_true')
parser.add_argument('--p1_lr', type=float, default=5e-5)
parser.add_argument('--p2_lr', type=float, default=2e-4)
parser.add_argument('--disc_lr', type=float, default=1e-3)
parser.add_argument('--repr_dim', type=int, default=64)
parser.add_argument('--lambda_1', type=float, default=20.0) # P1 -> disc
parser.add_argument('--lambda_2', type=float, default=5.0) # P1 -> P2
parser.add_argument('--lambda_3', type=float, default=20.0) # P2 -> disc
parser.add_argument('--batch_size', type=float, default=256)
args = parser.parse_args()

if args.randomize_hparams:
    args.p1_lr     = float(10**np.random.uniform(-5, -3))
    args.p2_lr     = float(10**np.random.uniform(-5, -3))
    args.disc_lr   = float(10**np.random.uniform(-5, -3))
    args.repr_dim  = int(2**np.random.uniform(2, 9))
    args.lambda_1  = float(10**np.random.uniform(-2, 2))
    args.lambda_2  = float(10**np.random.uniform(-2, 2))
    args.lambda_3  = float(10**np.random.uniform(-2, 2))

print('Args:')
for k,v in sorted(vars(args).items()):
    print(f'\t{k}: {v}')

X_source, y_source, X_target, y_target = datasets.colored_mnist()
X_target, y_target, X_test, y_test = datasets.split(X_target, y_target, 0.9)

p1_rep = ops.MultipleLinear(2*784, args.repr_dim, N_INSTANCES).cuda()
p1_classifier = ops.MultipleLinear(args.repr_dim, 10, N_INSTANCES).cuda()

p2_rep = ops.MultipleLinear(2*784, args.repr_dim, N_INSTANCES).cuda()

def make_disc():
    return nn.Sequential(
        ops.MultipleLinear(args.repr_dim, 256, N_INSTANCES),
        nn.ReLU(),
        ops.MultipleLinear(256, 256, N_INSTANCES),
        nn.ReLU(),
        ops.MultipleLinear(256, 1, N_INSTANCES)
    ).cuda()
disc_1 = make_disc()
disc_2 = make_disc()

p1_opt = optim.Adam(
    list(p1_rep.parameters()) + list(p1_classifier.parameters()),
    lr=args.p1_lr, betas=(0., 0.99)
)

p2_opt = optim.Adam(
    list(p2_rep.parameters()),
    lr=args.p2_lr, betas=(0., 0.99)
)

disc_opt = optim.Adam(
    list(disc_1.parameters()) + list(disc_2.parameters()),
    lr=args.disc_lr, betas=(0., 0.99)
)

def forward():
    xs, ys = ops.get_batch([X_source, y_source], N_INSTANCES*args.batch_size)
    xt = ops.get_batch(X_target, N_INSTANCES*args.batch_size)
    xs = xs.view(N_INSTANCES, args.batch_size, xs.shape[1])
    xt = xt.view(N_INSTANCES, args.batch_size, xt.shape[1])
    ys = ys.view(N_INSTANCES, args.batch_size)

    p1_rep_s = p1_rep(xs)
    p1_rep_t = p1_rep(xt)

    p2_rep_t = p2_rep(xt)

    disc_loss = F.binary_cross_entropy_with_logits(
        disc_1(p1_rep_s)[:,:,0], torch.ones_like(p1_rep_s[:,:,0])
    ) + F.binary_cross_entropy_with_logits(
        disc_1(p1_rep_t)[:,:,0], torch.zeros_like(p1_rep_t[:,:,0])
    ) + F.binary_cross_entropy_with_logits(
        disc_2(p1_rep_s)[:,:,0], torch.ones_like(p1_rep_s[:,:,0])
    ) + F.binary_cross_entropy_with_logits(
        disc_2(p2_rep_t)[:,:,0], torch.zeros_like(p2_rep_t[:,:,0])
    )
    disc_loss /= 4.

    classifier_loss = F.cross_entropy(p1_classifier(p1_rep_s).permute(0,2,1),ys)

    p2_loss = -(
          F.softmax(p1_classifier(p1_rep_t), dim=2)
        - F.softmax(p1_classifier(p2_rep_t), dim=2)
    ).abs().sum(dim=2).mean()

    return classifier_loss, disc_loss, p2_loss

utils.print_row(
    'step',
    'classifier loss',
    'disc loss',
    'p2 loss',
    'min test acc',
    'med test acc',
    'max test acc'
)

classifier_losses = []
disc_losses = []
p2_losses = []
for step in range(20001):
    # D step
    classifier_loss, disc_loss, p2_loss = forward()
    p1_opt.zero_grad()
    p2_opt.zero_grad()
    disc_opt.zero_grad()
    disc_loss.backward()
    disc_opt.step()

    # P1 step
    classifier_loss, disc_loss, p2_loss = forward()
    p1_opt.zero_grad()
    p2_opt.zero_grad()
    disc_opt.zero_grad()
    (classifier_loss
        - (args.lambda_1 * disc_loss) 
        - (args.lambda_2 * p2_loss)).backward()
    p1_opt.step()

    # P2 step
    classifier_loss, disc_loss, p2_loss = forward()
    p1_opt.zero_grad()
    p2_opt.zero_grad()
    disc_opt.zero_grad()
    (p2_loss - (args.lambda_3 * disc_loss)).backward()
    p2_opt.step()

    classifier_losses.append(classifier_loss.item())
    disc_losses.append(disc_loss.item())
    p2_losses.append(p2_loss.item())

    if step % 1000 == 0:
        with torch.no_grad():
            x = torch.stack(N_INSTANCES*[X_test], dim=0)
            logits = p1_classifier(p1_rep(x.cuda()))
            test_accs = []
            for i in range(N_INSTANCES):
                test_accs.append(
                    ops.multiclass_accuracy(logits[i,:,:], y_test).mean().item()
                )
        utils.print_row(
            step,
            np.mean(classifier_losses),
            np.mean(disc_losses),
            np.mean(p2_losses),
            np.min(test_accs),
            np.median(test_accs),
            np.max(test_accs)
        )
        classifier_losses, disc_losses, p2_losses = [], [], []