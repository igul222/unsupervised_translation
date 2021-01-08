import collections
import numpy as np
import lib
import time
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd

def get_X_batch(X, n_instances, batch_size):
    X = lib.ops.get_batch([X], n_instances*batch_size)[0]
    X = X.view(n_instances, batch_size, -1)
    return X

def get_Xy_batch(X, y, n_instances, batch_size):
    X, y = lib.ops.get_batch([X, y], n_instances*batch_size)
    X = X.view(n_instances, batch_size, -1)
    y = y.view(n_instances, batch_size)
    return X, y

def calculate_divergences(Zs, Zt):
    return torch.stack([
        lib.energy_dist.batched_energy_dist(Zs[i], Zt[i])
        for i in range(Zs.shape[0])
    ])

def make_disc(dim_in, disc_dim, n_instances):
    return lib.networks.multi_mlp(n_instances, dim_in, 1, disc_dim, 2)

def gan_loss_and_gp(Zs, Zt, disc, detach_Zs):
    # Gradient reversal trick
    if detach_Zs:
        Zs = Zs.detach().requires_grad_()
    else:
        Zs = (Zs.detach()*2) - Zs
    Zt = (Zt.detach()*2) - Zt

    disc_s = disc(Zs)
    disc_t = disc(Zt)
    disc_out = torch.cat([disc_s, disc_t], dim=1)

    targets = torch.zeros((1, 2*Zs.shape[1], 1), device='cuda')
    targets[:, :Zs.shape[1], :] += 1
    targets = targets.expand(Zs.shape[0], -1, -1)
    disc_loss = F.binary_cross_entropy_with_logits(disc_out, targets,
        reduction='none').mean(dim=[1,2]).sum()

    grad = autograd.grad([disc_out.sum()], [Zs], create_graph=True)[0]
    grad_penalty = grad.square().sum(dim=2).mean(dim=1).sum(dim=0)

    return disc_loss, grad_penalty

def wgangp_loss_and_gp(Zs, Zt, disc):
    # Gradient reversal trick
    Zs = (Zs.detach()*2) - Zs
    Zt = (Zt.detach()*2) - Zt

    disc_s = disc(Zs)
    disc_t = disc(Zt)

    disc_loss = disc_s.mean() - disc_t.mean()

    alpha = torch.rand((Zs.shape[0], Zs.shape[1], 1), device='cuda')
    interps = ((alpha*Zs) + ((1-alpha)*Zt)).detach()
    interps.requires_grad = True
    disc_interps = disc(interps)
    grad = autograd.grad(disc_interps.sum(), [interps], create_graph=True)[0]
    grad_penalty = (grad.norm(p=2, dim=2) - 1).pow(2).mean()

    return disc_loss, grad_penalty

def calculate_orth_penalty(W):
    eye = torch.eye(W.shape[1], device='cuda')[None,:,:]
    WWT = torch.bmm(W, W.permute(0,2,1))
    return (WWT - eye).square().sum(dim=[1,2]).sum()

OPT_BETAS = {
    'gan': (0.5, 0.99),
    'wgan-gp': (0., 0.99)
}

LOSS_AND_GP_FNS = {
    'gan': gan_loss_and_gp,
    'wgan-gp': wgangp_loss_and_gp
}

def train_dann(
    X_source, y_source, X_target, y_target, n_instances,
    rep_network,
    batch_size,
    detach_Zs,
    disc_dim,
    l2reg_c,
    l2reg_d,
    l2reg_r,
    lambda_erm,
    lambda_gp,
    lambda_orth,
    lr_d,
    lr_g,
    steps,
    z_dim):

    print('DANN:')

    if rep_network == 'linear':
        source_rep = lib.ops.MultipleLinear(X_source.shape[1], z_dim,
            n_instances, bias=False).cuda()
        target_rep = lib.ops.MultipleLinear(X_target.shape[1], z_dim,
            n_instances, bias=False).cuda()
    elif rep_network == 'mlp':
        source_rep = lib.networks.multi_mlp(n_instances, X_source.shape[1],
            z_dim, 128, 1)
        target_rep = lib.networks.multi_mlp(n_instances, X_target.shape[1],
            z_dim, 128, 1)
    elif rep_network == 'cnn':
        source_rep = lib.networks.MultiCNN(n_instances).cuda()
        target_rep = lib.networks.MultiCNN(n_instances).cuda()
        z_dim = 32*7*7

    classifier = lib.networks.multi_mlp(n_instances, z_dim,
        int(y_source.max()+1), 128, 1)
    disc = make_disc(z_dim, disc_dim, n_instances)

    rep_params = list(source_rep.parameters()) + list(target_rep.parameters())
    opt = optim.Adam([{
            'params': rep_params,
            'lr': lr_g,
            'weight_decay': l2reg_r
        }, {
            'params': classifier.parameters(),
            'lr': lr_g,
            'weight_decay': l2reg_c
        }, {
            'params': disc.parameters(),
            'lr': lr_d,
            'weight_decay': l2reg_d
        }], betas=OPT_BETAS['gan'])

    def forward():
        Xs, ys = get_Xy_batch(X_source, y_source, n_instances, batch_size)
        Xt, yt = get_Xy_batch(X_target, y_target, n_instances, batch_size)
        Zs = source_rep(Xs)
        Zt = target_rep(Xt)

        disc_loss, grad_penalty = gan_loss_and_gp(Zs, Zt, disc, detach_Zs)
        erm_loss = F.cross_entropy(classifier(Zs).permute(0,2,1), ys,
            reduction='none').mean(dim=1).sum(dim=0)
        if rep_network == 'linear':
            orth_penalty = (calculate_orth_penalty(source_rep.weight)
                + calculate_orth_penalty(target_rep.weight))
        else:
            orth_penalty = torch.tensor(0., device='cuda')
        with torch.no_grad():
            energy_dist = lib.energy_dist.energy_dist(Zs, Zt, unbiased=True)
            energy_dist = energy_dist.mean()

        source_acc = lib.ops.multiclass_accuracy(classifier(Zs), ys).mean()
        target_acc = lib.ops.multiclass_accuracy(classifier(Zt), yt).mean()

        loss = (disc_loss
            + (lambda_erm * erm_loss)
            + (lambda_gp * grad_penalty)
            + (lambda_orth * orth_penalty)
        )

        ni = float(n_instances)
        return (loss, disc_loss/ni, erm_loss/ni, grad_penalty/ni,
            orth_penalty/ni, energy_dist, source_acc, target_acc)

    lib.utils.train_loop(
        forward, opt, steps,
        history_names=['disc_loss', 'erm_loss', 'grad_penalty', 'orth_penalty',
            'energy_dist', 'source_acc', 'target_acc'], print_freq=100)

    with torch.no_grad():
        Xs = X_source[None,:,:].expand(n_instances,-1,-1)
        Xt = X_target[None,:,:].expand(n_instances,-1,-1)
        ys = y_source[None,:].expand(n_instances,-1)
        yt = y_target[None,:].expand(n_instances,-1)
        # Temporarily move Zs off of GPU memory while computing Zt
        Zs = source_rep(Xs).cpu()
        Zt = target_rep(Xt)
        Zs = Zs.cuda()
        source_accs = lib.ops.multiclass_accuracy(
            classifier(Zs), ys).mean(dim=1)
        target_accs = lib.ops.multiclass_accuracy(
            classifier(Zt), yt).mean(dim=1)
    divergences = calculate_divergences(Zs, Zt)

    lib.utils.print_row('instance', 'divergence', 'source acc', 'target acc')
    for idx in torch.argsort(divergences):
        lib.utils.print_row(idx, divergences[idx], source_accs[idx],
            target_accs[idx])

    return source_rep, target_rep, classifier, divergences, target_accs