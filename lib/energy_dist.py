import torch

def offdiag_mean(M):
    """
    Given a matrix or batch of matrices, return the mean (or means) of all
    off-diagonal entries.
    """
    if len(M.shape) == 2:
        M = M[None, :, :]
    N = M.shape[1]
    M_reshaped = M.view(-1, N*N)[:, :-1].view(-1, N-1, N+1)
    result = M_reshaped[:, :, 1:].mean(dim=[-2, -1])
    if len(M.shape) == 2:
        result = result[0]
    return result

@torch.jit.script
def _energy_dist(X, Y, unbiased: bool):
    X = X.half()
    Y = Y.half()
    DXX = torch.cdist(X, X)
    DXY = torch.cdist(X, Y)
    DYY = torch.cdist(Y, Y)
    M = (DXY - DXX + DXY - DYY)
    if unbiased:
        return offdiag_mean(M)
    else:
        return M.mean(dim=[-2, -1])

def energy_dist(X, Y, unbiased=False):
    with torch.cuda.amp.autocast(enabled=False):
        return _energy_dist(X, Y, unbiased)

@torch.jit.script
def _batch_energy_dist(X1, X2, Y1, Y2):
    DXX = torch.cdist(X1, X2)
    DXY = torch.cdist(X1, Y2)
    DYY = torch.cdist(Y1, Y2)
    return (DXY - DXX + DXY - DYY).float().mean(dim=[-2,-1])

def batched_energy_dist(X, Y):
    BS = 16384
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            X = X.half()
            Y = Y.half()
            block_dists = []
            block_weights = []
            for i in range(0, X.shape[0], BS):
                for j in range(0, Y.shape[0], BS):
                    X1, X2 = X[i:i+BS], X[j:j+BS]
                    Y1, Y2 = Y[i:i+BS], Y[j:j+BS]
                    block_dists.append(_batch_energy_dist(X1,X2,Y1,Y2))
                    block_weights.append((X1.shape[0]/BS) * (X2.shape[0]/BS))
        block_dists = torch.stack(block_dists, dim=0)
        block_weights = torch.tensor(block_weights, device='cuda')
        return (block_dists * block_weights).mean() / block_weights.mean()