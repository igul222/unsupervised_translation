"""
DANN baseline: train many independent DANN models and report min/median/max
test accuracy over the models.

Best Colored MNIST result is 0.36 median / 0.87 max test acc, with default
hparams. Hparams found by random search and a fine-grained random search around
the best point, 20 trials each.
"""

import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import lib
from lib import ops, utils, datasets, pca
import argparse
import collections

parser = argparse.ArgumentParser()
parser.add_argument('--randomize_hparams', action='store_true')
parser.add_argument('--pca', action='store_true')
parser.add_argument('--pca_dim', type=int, default=256)
parser.add_argument('--disc_lr', type=float, default=5e-4)
parser.add_argument('--gen_lr', type=float, default=1e-4)
parser.add_argument('--repr_dim', type=int, default=64)
parser.add_argument('--lambda_erm', type=float, default=0.1)
parser.add_argument('--lambda_gp', type=float, default=1.0)
parser.add_argument('--lambda_orth', type=float, default=0.0)
parser.add_argument('--lambda_cov', type=float, default=10.0)
parser.add_argument('--lambda_skew', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--n_instances', type=int, default=8)
args = parser.parse_args()


if args.randomize_hparams:
    args.disc_lr        = float(10**np.random.uniform(-5, -3))
    args.gen_lr         = float(10**np.random.uniform(-5, -3))
    args.repr_dim       = int(2**np.random.uniform(2, 9))
    args.lambda_erm     = float(10**np.random.uniform(-2, 2))
    args.lambda_gp      = float(10**np.random.uniform(-2, 1))

print('Args:')
for k,v in sorted(vars(args).items()):
    print(f'\t{k}: {v}')

X_source, y_source, X_target, y_target = datasets.colored_mnist()
X_target, y_target, X_test, y_test = datasets.split(X_target, y_target, 0.9)

if args.pca:
    source_pca = pca.PCA(X_source, args.pca_dim, whiten=True)
    target_pca = pca.PCA(X_target, args.pca_dim, whiten=True)
    X_source = source_pca.forward(X_source)
    X_target = target_pca.forward(X_target)
    X_test = target_pca.forward(X_test)

# Apply random orthogonal transforms for optimization reasons.
W1 = ops.random_orthogonal_matrix(X_source.shape[1])
W2 = ops.random_orthogonal_matrix(X_target.shape[1])
X_source = X_source @ W1.T
X_target = X_target @ W2.T
X_test   = X_test   @ W2.T

source_rep = ops.MultipleLinear(X_source.shape[1], args.repr_dim, args.n_instances, bias=False).cuda()
target_rep = ops.MultipleLinear(X_target.shape[1], args.repr_dim, args.n_instances, bias=False).cuda()
classifier = ops.MultipleLinear(args.repr_dim, 10, args.n_instances).cuda()
disc = nn.Sequential(
    ops.MultipleLinear(args.repr_dim, 512, args.n_instances), nn.GELU(),
    ops.MultipleLinear(512, 512, args.n_instances), nn.GELU(),
    ops.MultipleLinear(512, 1, args.n_instances)
).cuda()

gen_params = (
    list(source_rep.parameters()) + 
    list(target_rep.parameters()) +
    list(classifier.parameters())
)
opt = optim.Adam([
    {'params': gen_params, 'lr': args.gen_lr, 'betas': (0., 0.99)},
    {'params': disc.parameters(), 'lr': args.disc_lr, 'betas': (0., 0.99)}
])

disc_targets = torch.zeros((1, 2*args.batch_size, 1), device='cuda')
disc_targets[:, :args.batch_size, :] += 1
disc_targets = disc_targets.expand(args.n_instances, -1, -1)

eye = torch.eye(args.repr_dim, device='cuda')[None,:,:]

def forward():
    xs, ys = ops.get_batch([X_source, y_source], args.n_instances*args.batch_size)
    xt = ops.get_batch(X_target, args.n_instances*args.batch_size)
    xs = xs.view(args.n_instances, args.batch_size, xs.shape[1])
    xt = xt.view(args.n_instances, args.batch_size, xt.shape[1])
    ys = ys.view(args.n_instances, args.batch_size)

    # source_rep and target_rep have unit-norm rows, which makes zs and zt tiny.
    # We rescale them to fix this.
    zs = source_rep(xs)# * float(np.sqrt(X_source.shape[1]/args.repr_dim))
    zt = target_rep(xt)# * float(np.sqrt(X_target.shape[1]/args.repr_dim))

    # Gradient reversal trick
    zs_reverse_grad = (zs.detach()*2) - zs
    zt_reverse_grad = (zt.detach()*2) - zt

    disc_in = torch.cat([zs_reverse_grad, zt_reverse_grad], dim=1)
    disc_out = disc(disc_in)
    disc_loss = F.binary_cross_entropy_with_logits(disc_out, disc_targets)

    grad_s = autograd.grad(disc_out.sum(), [disc_in], create_graph=True)[0]
    grad_penalty = grad_s.square().sum(dim=2).mean()
    disc_loss = disc_loss + (args.lambda_gp*grad_penalty)

    erm_loss = F.cross_entropy(classifier(zs).permute(0,2,1), ys)

    W = torch.randn((args.n_instances, 128, args.repr_dim), device='cuda')
    W = W / W.norm(p=2, dim=2, keepdim=True)
    Wzs = torch.bmm(zs, W.permute(0,2,1))
    cov_penalty = (Wzs.var(dim=1) - 1).abs().mean()
    # cov_penalty_1 = Wzs[:, ::2].var(dim=1, unbiased=True) - 1
    # cov_penalty_2 = Wzs[:, 1::2].var(dim=1, unbiased=True) - 1
    # cov_penalty = (cov_penalty_1*cov_penalty_2).mean()

    # skew_1 = Wzs[:, ::2].pow(3)
    # skew_2 = Wzs[:, 1::2].pow(3)
    # skew_penalty = -torch.clamp(Wzs.pow(3).mean(dim=1).abs()
    skew_penalty = torch.tensor(0.).cuda()

    # skew_penalty = 1-torch.clamp(Wzs.pow(3).mean(dim=1).pow(2), max=1.0).mean()
    # skew_penalty = -(skew_penalty_1*skew_penalty_2).mean()

    # with torch.cuda.amp.autocast(enabled=False):
    #     zs1, zs2 = zs[:,::2].float(), zs[:,1::2].float()
    #     zs_cov_1 = torch.einsum('inx,iny->ixy', zs1, zs1) / float(zs1.shape[1]-1)
    #     zs_cov_2 = torch.einsum('inx,iny->ixy', zs2, zs2) / float(zs2.shape[1]-1)
    # orth_penalty = ((zs_cov_1 - eye)*(zs_cov_2-eye)).sum(dim=[1,2]).mean()

    W = source_rep.weight
    WWT = torch.einsum('iab,icb->iac', W, W)
    orth_penalty = (WWT - eye).pow(2).sum(dim=[1,2]).mean()

    loss = (
        disc_loss
        + (args.lambda_erm * erm_loss)
        + (args.lambda_gp * grad_penalty)
        + (args.lambda_orth * orth_penalty)
        + (args.lambda_cov * cov_penalty)
        + (args.lambda_skew * skew_penalty)
    )

    return loss, erm_loss, disc_loss, grad_penalty, orth_penalty, cov_penalty, skew_penalty

utils.print_row(
    'step',
    'erm loss',
    'disc loss',
    'grad penalty',
    'orth penalty',
    'cov penalty',
    'skew penalty',
    'min test acc',
    'med test acc',
    'max test acc'
)

scaler = torch.cuda.amp.GradScaler()
histories = collections.defaultdict(lambda: [])
for step in range(10001):
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        loss, erm_loss, disc_loss, grad_penalty, orth_penalty, cov_penalty, skew_penalty = forward()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    histories['erm_loss'].append(erm_loss.item())
    histories['disc_loss'].append(disc_loss.item())
    histories['grad_penalty'].append(grad_penalty.item())
    histories['orth_penalty'].append(orth_penalty.item())
    histories['cov_penalty'].append(cov_penalty.item())
    histories['skew_penalty'].append(skew_penalty.item())

    if step % 1000 == 0:
        with torch.no_grad():
            x = torch.stack(args.n_instances*[X_test], dim=0)
            logits = classifier(
                target_rep(x) * float(np.sqrt(X_target.shape[1]/args.repr_dim))
            )
            test_accs = [
                ops.multiclass_accuracy(logits[i,:,:], y_test).mean().item()
                for i in range(args.n_instances)
            ]
        utils.print_row(
            step,
            np.mean(histories['erm_loss']),
            np.mean(histories['disc_loss']),
            np.mean(histories['grad_penalty']),
            np.mean(histories['orth_penalty']),
            np.mean(histories['cov_penalty']),
            np.mean(histories['skew_penalty']),
            np.min(test_accs),
            np.median(test_accs),
            np.max(test_accs)
        )
        histories.clear()