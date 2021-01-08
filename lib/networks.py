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

class MultipleConv(nn.Module):
    def __init__(self, n_instances, dim_in, dim_out, kernel_size):
        super().__init__()
        self.dim_out = dim_out
        self._conv = nn.Conv2d(
            in_channels=n_instances*dim_in,
            out_channels=n_instances*dim_out,
            kernel_size=kernel_size,
            padding=(kernel_size-1)//2,
            stride=2,
            groups=n_instances)
    def forward(self,x):
        bs,n,c_in,h,w = x.shape
        x = x.view((bs, n*c_in, h, w))
        x = self._conv(x)
        h, w = x.shape[2:]
        x = x.view((bs, n, self.dim_out, h, w))
        return x

class MultiCNN(nn.Module):
    def __init__(self, n_instances):
        super().__init__()
        self.conv1 = MultipleConv(n_instances, 2, 16, 5)
        self.conv2 = MultipleConv(n_instances, 16, 32, 5)
    def forward(self,x):
        assert(x.shape[2] == 2*784)
        x = x.view(x.shape[0], x.shape[1], 28, 28, 2)
        n, b, h, w, c = x.shape
        x = x.permute(1,0,4,2,3).contiguous()
        x = self.conv1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = x.permute(1,0,3,4,2).contiguous()
        x = x.view(n, b, h*w*c)
        return x