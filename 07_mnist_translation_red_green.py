"""
Red MNIST to green MNIST
"""

import numpy as np
import torch
from torch import nn, optim, autograd
from torchvision import datasets
import torch.nn.functional as F
import lib

N_TRAIN = 60000
BATCH_SIZE = 512
DIM = 256
WGANGP_LAMDA = 10.
LR = 5e-4

mnist_red_tr, mnist_green_tr, mnist_red_va, mnist_green_va = \
    lib.make_red_and_green_mnist()

def init_model():
    generator = nn.Linear(2*196, 2*196).cuda()
    discriminator = nn.Sequential(
        nn.Linear(2*196, DIM),
        nn.ReLU(),
        nn.Linear(DIM, DIM),
        nn.ReLU(),
        nn.Linear(DIM, 1)
    ).cuda()

    gen_opt = optim.Adam(generator.parameters(), lr=LR, betas=(0., 0.99))
    disc_opt = optim.Adam(discriminator.parameters(), lr=LR, betas=(0., 0.99))
    return generator, discriminator, gen_opt, disc_opt

def l2(params):
    return torch.cat([p.pow(2).view(-1) for p in params]).sum()

def forward(detach=False):
    x_real = lib.get_batch(mnist_green_tr, BATCH_SIZE)
    x_fake = generator(lib.get_batch(mnist_red_tr, BATCH_SIZE))

    disc_real = discriminator(x_real)
    disc_fake = discriminator(x_fake)
    loss = disc_real.mean() - disc_fake.mean()
    epsilon = torch.rand(BATCH_SIZE, 1).cuda()
    interps = (epsilon*x_real) + ((1-epsilon)*x_fake)
    disc_interps = discriminator(interps)
    grad = autograd.grad(disc_interps.sum(), [interps], create_graph=True)[0]
    grad_norm = (grad.pow(2).sum(dim=1) + 1e-6).sqrt()
    gp = (grad_norm - 1).pow(2).mean()
    return loss + (WGANGP_LAMDA * gp)

def l2_eval():
    EVAL_BS = 4096
    x_red = lib.get_batch(mnist_red_tr, EVAL_BS)
    x_fake = generator(x_red)
    x_real = x_red.view(EVAL_BS, 14, 14, 2).flip(3).view(EVAL_BS, 2*196)
    return (x_fake - x_real).pow(2).mean()

for restart in range(10):
    generator, discriminator, gen_opt, disc_opt = init_model()
    lib.print_row('step', 'loss', 'l2_eval')
    loss_vals = []
    for step in range(5001):
        for inner_step in range(5):
            loss = forward(detach=True)
            gen_opt.zero_grad()
            disc_opt.zero_grad()
            loss.backward()
            disc_opt.step()
        loss = forward()
        gen_opt.zero_grad()
        disc_opt.zero_grad()
        (-loss).backward()
        gen_opt.step()
        loss_vals.append(loss.item())
        if step % 1000 == 0:
            lib.print_row(step, np.mean(loss_vals), l2_eval())
            loss_vals = []
            lib.save_image_grid_colored_mnist(mnist_red_tr[:100].cpu().numpy(),
                f'outputs/07_restart{restart}_original.png')
            lib.save_image_grid_colored_mnist(
                generator(mnist_red_tr[:100]).cpu().detach().numpy(),
                f'outputs/07_restart{restart}_translated.png'
            )