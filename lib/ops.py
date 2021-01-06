import torch
import torch.nn.functional as F
from torch import nn

def wasserstein_1d(p, q):
    """Computes W_1 between two 1-dimensional distributions"""
    p, _ = torch.sort(p)
    q, _ = torch.sort(q)
    return torch.abs(p - q).mean()

def get_batch(vars_, batch_size):
    assert(isinstance(vars_, list))
    idx = torch.randint(low=0, high=len(vars_[0]), size=(batch_size,))
    return [v[idx] for v in vars_]

def multiclass_accuracy(y_pred, y):
    return torch.argmax(y_pred, dim=-1).eq(y).float()

def binary_accuracy(logits, y):
    assert(logits.shape == y.shape)
    return 1 - torch.abs((logits > 0).float() - y).mean()

def distance_matrix(X, Y):
    BS = 16384
    M = torch.zeros((X.shape[0], Y.shape[0])).cuda()
    for i in range(0, X.shape[0], BS):
        for j in range(0, Y.shape[0], BS):
            X1 = X[i:i+BS]
            X2 = Y[j:j+BS]
            M[i:i+BS, j:j+BS] = torch.cdist(X1, X2)
    return M

def nearest_neighbors(X, v):
    """Returns indices of rows of X in descending order of cosine dist to v."""
    X_normalized = X / (1e-8 + X.norm(p=2, dim=1, keepdim=True))
    dists = X_normalized @ v
    return torch.flip(torch.argsort(dists), [0])

def random_orthogonal_matrix(N):
    W = torch.zeros((N, N), device='cuda')
    torch.nn.init.orthogonal_(W)
    return W

class MultipleLinear(nn.Module):
    """
    A collection of linear transformations that can be applied batchwise.
    Useful for efficiently training many neural nets in parallel (e.g. many 
    random seeds).
    """
    def __init__(self, dim_in, dim_out, n_instances, bias=True,
        init='orthogonal'):
        
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((n_instances, dim_out, dim_in)))
        for i in range(n_instances):
            if init == 'orthogonal':
                torch.nn.init.orthogonal_(self.weight.data[i])
            elif init == 'xavier':
                torch.nn.init.xavier_uniform_(self.weight.data[i])
            else:
                raise Exception()
        if bias:
            self.bias = nn.Parameter(torch.zeros((n_instances, dim_out)))
        else:
            self.bias = None
    def forward(self, x):
        if self.bias is not None:
            x = torch.baddbmm(self.bias[:,None,:], x,
                self.weight.permute(0, 2, 1))
        else:
            x = torch.bmm(x, self.weight.permute(0, 2, 1))
        return x

def softmax_cross_entropy(logits, targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1)