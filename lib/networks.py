"""
Neural net architecture implementations.
"""

import functools
import lib
import torch
from torch import nn

def multi_linear(n_instances, dim_in, dim_out):
    return lib.ops.MultipleLinear(dim_in, dim_out, n_instances,
        bias=False).cuda()

def multi_mlp(n_instances, dim_in, dim_out, dim_hid, n_hid):
    assert(n_hid >= 1)
    linear = functools.partial(lib.ops.MultipleLinear, n_instances=n_instances,
        init='xavier')
    layers = []
    layers.append(linear(dim_in, dim_hid))
    layers.append(nn.ReLU())
    for _ in range(n_hid-1):
        layers.append(linear(dim_hid, dim_hid))
        layers.append(nn.ReLU())
    layers.append(linear(dim_hid, dim_out))
    return nn.Sequential(*layers).cuda()