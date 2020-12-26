import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import lib
from lib import ops, utils, datasets, pca
import argparse
import collections

parser = argparse.ArgumentParser()
parser.add_argument('--pca', action='store_false')
parser.add_argument('--pca_dim', type=int, default=256)
parser.add_argument('--disc_lr', type=float, default=5e-4)
parser.add_argument('--gen_lr', type=float, default=5e-4)
parser.add_argument('--repr_dim', type=int, default=64)
parser.add_argument('--lambda_erm', type=float, default=0.1)
parser.add_argument('--lambda_gp', type=float, default=1.0)
parser.add_argument('--lambda_cov', type=float, default=10.0)
parser.add_argument('--lambda_skewness', type=float, default=0.0)
parser.add_argument('--n_instances', type=int, default=1)
args = parser.parse_args()

print('Args:')
for k,v in sorted(vars(args).items()):
    print(f'\t{k}: {v}')

X_source, y_source, X_target, y_target = datasets.colored_mnist()

if args.pca:
    source_pca = pca.PCA(X_source, args.pca_dim, whiten=True)
    target_pca = pca.PCA(X_target, args.pca_dim, whiten=True)
    X_source = source_pca.forward(X_source)
    X_target = target_pca.forward(X_target)
else:
    X_source_mean = X_source.mean(dim=0, keepdim=True)
    X_target_mean = X_target.mean(dim=0, keepdim=True)
    X_source -= X_source_mean
    X_target -= X_target_mean

# Apply random orthogonal transforms for optimization reasons.
W1 = ops.random_orthogonal_matrix(X_source.shape[1])
W2 = ops.random_orthogonal_matrix(X_target.shape[1])
X_source = X_source @ W1.T
X_target = X_target @ W2.T

source_rep = ops.MultipleLinear(X_source.shape[1], args.repr_dim,
    args.n_instances, bias=False).cuda()
target_rep = ops.MultipleLinear(X_target.shape[1], args.repr_dim,
    args.n_instances, bias=False).cuda()
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

disc_targets = torch.zeros((1, X_source.shape[0]+X_target.shape[0], 1),
    device='cuda')
disc_targets[:, :X_source.shape[0], :] += 1
disc_targets = disc_targets.expand(args.n_instances, -1, -1)

eye = torch.eye(args.repr_dim, device='cuda')[None,:,:]

def forward():
    xs = X_source[None,:,:].expand(args.n_instances, -1, -1)
    xt = X_target[None,:,:].expand(args.n_instances, -1, -1)
    ys = y_source[None,:].expand(args.n_instances, -1)

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

    with torch.cuda.amp.autocast(enabled=False):
        zs_ = zs.float()
        zs_cov = torch.einsum('inx,iny->ixy', zs_, zs_) / float(zs_.shape[1])
    cov_penalty = (zs_cov - eye).pow(2).sum(dim=[1,2]).mean()

    W = torch.randn((args.n_instances, 128, args.repr_dim), device='cuda')
    W = W / W.norm(p=2, dim=2, keepdim=True)
    Wzs = torch.bmm(zs, W.permute(0,2,1))
    skewness = Wzs.pow(3).mean(dim=1).abs()
    skewness_penalty = 1 - torch.clamp(skewness, max=1.0).mean()

    loss = (
        disc_loss
        + (args.lambda_erm * erm_loss)
        + (args.lambda_gp * grad_penalty)
        + (args.lambda_cov * cov_penalty)
        + (args.lambda_skewness * skewness_penalty)
    )

    return loss, erm_loss, disc_loss, grad_penalty, cov_penalty, skewness_penalty

utils.print_row(
    'step',
    'erm loss',
    'disc loss',
    'grad penalty',
    'cov penalty',
    'skewness penalty',
    'min test acc',
    'med test acc',
    'max test acc'
)

scaler = torch.cuda.amp.GradScaler()
histories = collections.defaultdict(lambda: [])
for step in range(10001):
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        loss, erm_loss, disc_loss, grad_penalty, cov_penalty, \
            skewness_penalty = forward()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    histories['erm_loss'].append(erm_loss.item())
    histories['disc_loss'].append(disc_loss.item())
    histories['grad_penalty'].append(grad_penalty.item())
    histories['cov_penalty'].append(cov_penalty.item())
    histories['skewness_penalty'].append(skewness_penalty.item())

    if step % 1000 == 0:
        print('Saving plots...')
        with torch.cuda.amp.autocast():
            import matplotlib
            import matplotlib.pyplot as plt
            xs = X_source[None,:,:].expand(args.n_instances, -1, -1)
            xt = X_target[None,:,:].expand(args.n_instances, -1, -1)
            ys = y_source[None,:].expand(args.n_instances, -1)
            yt = y_target[None,:].expand(args.n_instances, -1)

            zs = source_rep(xs)
            zt = target_rep(xt)

            def to_np(tensor):
                return tensor.cpu().detach().numpy()

            # Plot 1: both unlabeled distributions, overlaid
            plt.clf()
            plt.scatter(to_np(zs[0,:,0]), to_np(zs[0,:,1]))
            plt.scatter(to_np(zt[0,:,0]), to_np(zt[0,:,1]))
            plt.savefig(f'step{step}_both.png')

            # Plot 2: labeled source distribution
            plt.clf()
            plt.scatter(to_np(zs[0,:,0]), to_np(zs[0,:,1]), c=to_np(ys[0]),
                cmap='tab10')
            plt.savefig(f'step{step}_source.png')

            # Plot 3: labeled target distribution
            plt.clf()
            plt.scatter(to_np(zt[0,:,0]), to_np(zt[0,:,1]), c=to_np(yt[0]),
                cmap='tab10')
            plt.savefig(f'step{step}_target.png')

    if step % 100 == 0:
        with torch.no_grad():
            x = X_target[None,:,:].expand(args.n_instances, -1, -1)
            logits = classifier(target_rep(x))
            test_accs = [
                ops.multiclass_accuracy(logits[i,:,:], y_target).mean().item()
                for i in range(args.n_instances)
            ]
        utils.print_row(
            step,
            np.mean(histories['erm_loss']),
            np.mean(histories['disc_loss']),
            np.mean(histories['grad_penalty']),
            np.mean(histories['cov_penalty']),
            np.mean(histories['skewness_penalty']),
            np.min(test_accs),
            np.median(test_accs),
            np.max(test_accs)
        )
        histories.clear()