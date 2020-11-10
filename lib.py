import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets
import sys

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

def wasserstein_1d(p, q):
    """Computes W1 between two 1-dimensional distributions"""
    p, _ = torch.sort(p)
    q, _ = torch.sort(q)
    return torch.abs(p - q).mean()

def make_red_and_green_mnist():
    N_TRAIN = 50000
    mnist = datasets.MNIST('/tmp', train=True, download=True)
    rng_state = np.random.get_state()
    np.random.shuffle(mnist.data.numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist.targets.numpy())
    mnist_data = (mnist.data.reshape((60000, 28, 28)).float() / 256.).cuda()
    def apply_color(images, color):
        # 2x downsample for computational convenience
        images = (
            images.reshape((-1, 28, 28))[:, ::2, ::2] +
            images.reshape((-1, 28, 28))[:, 1::2, ::2] +
            images.reshape((-1, 28, 28))[:, ::2, 1::2] + 
            images.reshape((-1, 28, 28))[:, 1::2, 1::2]) / 4
        # apply color
        if color == 'red':
            images = torch.stack([images, torch.zeros_like(images)], dim=3)
        elif color == 'green':
            images = torch.stack([torch.zeros_like(images), images], dim=3)
        else:
            raise Exception()
        return images.reshape((-1, 2*196))
    mnist_red_tr = apply_color(mnist_data[:N_TRAIN:2], 'red')
    mnist_green_tr = apply_color(mnist_data[1:N_TRAIN:2], 'green')
    mnist_red_va = apply_color(mnist_data[N_TRAIN::2], 'red')
    mnist_green_va = apply_color(mnist_data[N_TRAIN+1::2], 'green')
    return mnist_red_tr, mnist_green_tr, mnist_red_va, mnist_green_va

def random_features_distance(x, y, n_feats):
    W = torch.randn((x.shape[1], n_feats), device='cuda')
    x_feats = F.relu(torch.matmul(x, W))
    y_feats = F.relu(torch.matmul(y, W))
    return (x_feats.mean(dim=0) - y_feats.mean(dim=0)).pow(2).mean()

def pairwise_distances(X, Y):
    return (X[:,None,:] - Y[None,:,:]).norm(p=2, dim=2)

def gaussian_kernel(X, Y):
    return torch.exp(-(pairwise_distances(X, Y)**2))

def energy_kernel(X, Y):
    return -pairwise_distances(X, Y)
 
def matrix_mean(d, triu):
    if triu:
        weights = torch.triu(torch.ones_like(d), diagonal=1)
        return torch.mean(d * weights) / torch.mean(weights)
    else:
        return torch.mean(d)
 
def mmd(kernel_fn, x, y, triu_x=True, triu_y=True):
    return (-2*matrix_mean(kernel_fn(x, y), False)
        + matrix_mean(kernel_fn(x, x), triu_x)
        + matrix_mean(kernel_fn(y, y), triu_y))

def get_batch(x, batch_size):
    idx = torch.randint(low=0, high=len(x), size=(batch_size,))
    return x[idx]

def print_row(*row, colwidth=10):
    def format_val(x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str( x).ljust(colwidth)[:colwidth]
    print("  ".join([format_val(x) for x in row]))

def save_image_grid(images, path):
  assert(images.ndim == 4) # BHWC
  assert(images.dtype == 'uint8')
  n_samples = images.shape[0]
  n_rows = int(np.sqrt(n_samples))
  while n_samples % n_rows != 0:
      n_rows -= 1
  n_cols = n_samples//n_rows
  # Copy each image into its spot in the grid
  height, width = images[0].shape[:2]
  grid_image = np.zeros((height*n_rows, width*n_cols, 3), dtype='uint8')
  for n, image in enumerate(images):
    j = n // n_cols
    i = n % n_cols
    grid_image[j*height:j*height+height, i*width:i*width+width] = image
  plt.imsave(path, grid_image)

def save_image_grid_mnist(samples, path):
    samples = samples.reshape((-1, 28, 28, 1))
    samples = np.repeat(samples, 3, axis=3)
    samples = np.clip(samples, 0.001, 0.999)
    samples = (samples * 256).astype('uint8')
    save_image_grid(samples, path)

def save_image_grid_colored_mnist(samples, path):
    samples = samples.reshape((-1, 14, 14, 2))
    samples = np.stack([
      samples[:,:,:,0],
      samples[:,:,:,1],
      np.zeros_like(samples[:,:,:,0])
      ], axis=3)
    samples = np.clip(samples, 0.001, 0.999)
    samples = (samples * 256).astype('uint8')
    save_image_grid(samples, path)

def multiclass_accuracy(y_pred, y):
    return torch.argmax(y_pred, dim=1).eq(y).float().mean()