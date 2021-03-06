import collections
import copy
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import torch
from torch import optim

def print_tensor(label, tensor):
    """Print a tensor with a given label."""
    torch.set_printoptions(precision=3, linewidth=119, sci_mode=False)
    print(f'{label}:')
    for line in str(tensor).splitlines():
        print(f"\t{line}")
    torch.set_printoptions(profile='default')

def print_row(*row, colwidth=16):
    """Print a row of values."""
    def format_val(x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str( x).ljust(colwidth)[:colwidth]
    print("  ".join([format_val(x) for x in row]))

def save_image_grid(images, path):
    """
    Given a tensor representing a batch of images, arrange them into a
    rectangular grid and save the images to the given path. The specific
    preprocessing is inferred based on the image shape, dtype, and values.
    Supported image formats:
    MNIST: float, shape (N, 784), values in [0, 1]
    Colored MNIST: float, shape (N, 2*784), channel-minor, values in [0, 1]
    All others: byte, shape (N, H, W, C), values in [0, 255]
    """
    assert(torch.is_tensor(images))

    if (images.shape[-1] == 784):
        # MNIST
        images = images.reshape((-1, 28, 28, 1))
        images = images.expand(-1, -1, -1, 3)
        images = images.clamp(min=0.001, max=0.999)
        images = (images * 256).byte()
    elif (images.shape[-1] == 2*784):
        # Colored MNIST
        images = images.reshape((-1, 28, 28, 2))
        images = torch.cat([images, torch.zeros_like(images[:,:,:,:1])], dim=3)
        images = images.clamp(min=0.001, max=0.999)
        images = (images * 256).byte()

    assert(images.ndim == 4) # BHWC
    assert(images.dtype == torch.uint8)
    images = images.detach().cpu().numpy()
    n_images = images.shape[0]
    n_rows = int(np.sqrt(n_images))
    while n_images % n_rows != 0:
        n_rows -= 1
    n_cols = n_images//n_rows
    # Copy each image into its spot in the grid
    height, width = images[0].shape[:2]
    grid_image = np.zeros((height*n_rows, width*n_cols, 3), dtype='uint8')
    for n, image in enumerate(images):
        j = n // n_cols
        i = n % n_cols
        grid_image[j*height:j*height+height, i*width:i*width+width] = image
    plt.imsave(path, grid_image)

def train_loop(forward, opt, steps, history_names=[], hook=None,
    print_freq=1000, quiet=False):

    if not quiet:
        print_row('step', 'step time', 'loss', *history_names)
    histories = collections.defaultdict(lambda: [])
    scaler = torch.cuda.amp.GradScaler()
    start_time = time.time()
    for step in range(steps):

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            forward_vals = forward()
            if not isinstance(forward_vals, tuple):
                forward_vals = (forward_vals,)
        scaler.scale(forward_vals[0]).backward()
        scaler.step(opt)
        scaler.update()
        # forward_vals[0].backward()
        # opt.step()

        histories['loss'].append(forward_vals[0].item())
        for name, val in zip(history_names, forward_vals[1:]):
            histories[name].append(val.item())

        if step % print_freq == 0:
            if not quiet:
                print_row(
                    step,
                    (time.time() - start_time) / (step+1),
                    np.mean(histories['loss']),
                    *[np.mean(histories[name]) for name in history_names]
                )
            if hook is not None:
                hook(step)
            histories.clear()

def enforce_orthogonality(X, threshold=0.1, verbose=False):
    """
    Project a matrix, or batch of matrices, onto the set of matrices with
    orthonormal rows. Assumes X has shape (m, n) or (b, m, n) with m <= n.
    """
    X_orig = X
    if len(X.shape) == 2:
        X = X[None,:,:]
    MAX_STEPS = 1000
    _eye_m = torch.eye(X.shape[1], device='cuda')[None,:,:]
    def forward():
        XXT = torch.einsum('iab,icb->iac', X, X)
        return torch.clamp((XXT - _eye_m).pow(2).sum(dim=[1,2]), min=threshold)
    opt = optim.SGD([X_orig], lr=1e-2)
    scaler = torch.cuda.amp.GradScaler()
    for step in range(MAX_STEPS):
        with torch.cuda.amp.autocast():
            losses = forward()
        opt.zero_grad()
        scaler.scale(losses.sum()).backward()
        scaler.step(opt)
        scaler.update()
        if verbose:
            print_row(step, losses.mean().item())
        if losses.max() < threshold + 1e-8:
            break
    if step == MAX_STEPS - 1:
        print("Warning: enforce_orthogonality didn't converge!")