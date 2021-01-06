import numpy as np
import os
import textwrap
import torch
import torch.nn.functional as F
import torchvision.datasets
import tqdm

DATA_DIR = os.path.expanduser('~/data')

def _parallel_shuffle(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def mnist():
    mnist = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True)
    X, y = mnist.data.clone(), mnist.targets.clone()
    _parallel_shuffle(X.numpy(), y.numpy())
    X = (X.float() / 256.)
    return X.view(-1, 784).cuda(), y.cuda()

def colored_mnist():
    X, y = mnist()
    X = X.view(-1, 28, 28)
    X_red, X_green = X[::2], X[1::2]
    X_red = torch.stack([X_red, 0*X_red], dim=3)
    X_green = torch.stack([0*X_green, X_green], dim=3)
    return X_red.view(-1, 2*784), y[::2], X_green.view(-1, 2*784), y[1::2]

def mnist_usps():
    X_source, y_source = mnist()
    usps = torchvision.datasets.USPS(DATA_DIR, train=True, download=True)
    X_target = torch.tensor(usps.data).clone()
    y_target = torch.tensor(usps.targets).clone()
    _parallel_shuffle(X_target.numpy(), y_target.numpy())
    X_target = (X_target.float() / 256.).cuda()
    y_target = y_target.cuda()
    # Prior work uses 2000 MNIST images, but various parts of this codebase
    # expect X_source and X_target to have the same number of images.
    X_source, y_source = X_source[:1800], y_source[:1800]
    X_target, y_target = X_target[:1800], y_target[:1800]
    X_target = F.interpolate(
        X_target.view(1800, 1, 16, 16),
        28,
        mode='bicubic',
        align_corners=False).view(1800, 784)
    return X_source, y_source, X_target, y_target

def binary_colored_mnist():
    """
    Like Colored MNIST, but the task is binary classification: digits 0-4
    form class 1, and 5-9 form class 0.
    """
    X, y = mnist()
    X = X.view(-1, 28, 28)
    X_red, X_green = X[::2], X[1::2]
    X_red = torch.stack([X_red, 0*X_red], dim=3)
    X_green = torch.stack([0*X_green, X_green], dim=3)
    y = (y < 5).long()
    return X_red.view(-1, 2*784), y[::2], X_green.view(-1, 2*784), y[1::2]

def _load_word_vectors(path):
    if not os.path.exists(path):
        error_message = textwrap.dedent(f"""\
        Couldn't find {path}!
        You probably want to download these to {DATA_DIR}:
        https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
        https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.vec
        """)
        raise Exception(error_message)
    processed_path = path + '.processed'
    if os.path.exists(processed_path):
        words, vectors = torch.load(processed_path)
        return words, vectors.cuda()
    else:
        words = []
        vectors = []
        with open(path, 'r') as f:
            for line_idx, line in tqdm.tqdm(enumerate(f)):
                if line_idx == 0:
                    continue # First line is some kind of header
                parts = line[:-2].split(' ')
                words.append(parts[0])
                vectors.append(torch.tensor([float(x) for x in parts[1:]]))
        vectors = torch.stack(vectors, dim=0)
        torch.save((words, vectors), processed_path)
        return words, vectors.cuda()

def en_word_vectors():
    return _load_word_vectors(os.path.join(DATA_DIR, 'wiki.en.vec'))

def de_word_vectors():
    return _load_word_vectors(os.path.join(DATA_DIR, 'wiki.de.vec'))

def _load_sentiment_lexicon(lang):
    path = os.path.join(DATA_DIR, f'positive_words_{lang}.txt')
    if not os.path.exists(path):
        raise Exception(textwrap.dedent(f"""\
            Couldn't find {path}!
            You probably want to download these to {DATA_DIR}:
            positive_words_en.txt
            negative_words_en.txt
            positive_words_de.txt
            negative_words_de.txt
            You can find these files at:
            https://sites.google.com/site/datascienceslab/projects/multilingualsentiment
            """))

    if lang == 'en':
        words, vectors = en_word_vectors()
    elif lang == 'de':
        words, vectors = de_word_vectors()

    n = 2000 # Total number of words to load
    lexicon_vectors = []
    lexicon_labels = []
    with open(os.path.join(DATA_DIR, f'positive_words_{lang}.txt'), 'r') as f:
        positive_words = [line[:-1] for line in f]
    with open(os.path.join(DATA_DIR, f'negative_words_{lang}.txt'), 'r') as f:
        negative_words = [line[:-1] for line in f]
    assert(len(positive_words) >= n//2)
    assert(len(negative_words) >= n//2)
    for word in positive_words[:n//2]:
        lexicon_vectors.append( vectors[words.index(word.lower())] )
        lexicon_labels.append(1)
    for word in negative_words[:n//2]:
        lexicon_vectors.append( vectors[words.index(word.lower())] )
        lexicon_labels.append(0)
    _parallel_shuffle(lexicon_vectors, lexicon_labels)

    return (
        torch.stack(lexicon_vectors, dim=0), 
        torch.tensor(lexicon_labels).float().cuda()
    )

def en_sentiment_lexicon():
    return _load_sentiment_lexicon('en')

def de_sentiment_lexicon():
    return _load_sentiment_lexicon('de')

def split(a, b, fraction):
    n = int(fraction * len(a))
    return a[:n], b[:n], a[n:], b[n:]

def irm_colored_mnist():
    mnist = torchvision.datasets.MNIST('~/data', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    def make_environment(images, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability
        # 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
        return {
            'images': (images.float()/255.).view(images.shape[0], 2*196).cuda(),
            'labels': labels[:, None].cuda()
        }

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.1),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.2),
        make_environment(mnist_val[0], mnist_val[1], 0.9)
    ]

    return envs

REGISTRY = {
    'colored_mnist': colored_mnist,
    'binary_colored_mnist': binary_colored_mnist,
    'mnist_usps': mnist_usps
}
