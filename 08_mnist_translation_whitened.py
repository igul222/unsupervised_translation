"""
Whitened MNIST identity translation
"""

import numpy as np
import torch
from torch import nn, optim, autograd
from torchvision import datasets
import torch.nn.functional as F
import lib
from sklearn.decomposition import PCA

N_TRAIN = 60000
BATCH_SIZE = 512
DIM = 256
WGANGP_LAMDA = 10.
LR = 5e-4
PCA_DIMS = 128

mnist = datasets.MNIST('/tmp', train=True, download=True)
rng_state = np.random.get_state()
np.random.shuffle(mnist.data.numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist.targets.numpy())
mnist_data = (mnist.data.reshape((60000, 784)).float() / 256.).cuda()

pca = PCA(n_components=PCA_DIMS, whiten=True)
pca.fit(mnist_data.clone().cpu().numpy())
mnist_whitened = torch.tensor(pca.transform(mnist_data.cpu().numpy())).cuda()
def inverse_pca(x):
    x = x.cpu().detach().numpy()
    x = pca.inverse_transform(x)
    x = torch.tensor(x).cuda()
    return x

def init_model():
    generator = nn.Linear(PCA_DIMS, PCA_DIMS).cuda()
    discriminator = nn.Sequential(
        nn.Linear(PCA_DIMS, DIM),
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
    x_real = lib.get_batch(mnist_whitened, BATCH_SIZE)
    x_fake = generator(lib.get_batch(mnist_whitened, BATCH_SIZE))

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
    x_real = lib.get_batch(mnist_whitened, 4096)
    x_fake = generator(x_real)
    return (inverse_pca(x_fake) - inverse_pca(x_real)).pow(2).mean()

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
            lib.save_image_grid_mnist(mnist_data[:100].cpu().numpy(),
                f'outputs/08_restart{restart}_original.png')
            lib.save_image_grid_mnist(
                inverse_pca(generator(mnist_whitened[:100])).cpu().detach().numpy(),
                f'outputs/08_restart{restart}_translated.png'
            )