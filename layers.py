import numpy as np
import torch
import torch.nn as nn
import math

from utils import *

def fft_product(X, W):
    Xf = torch.fft.fft(X)
    Wf = torch.fft.fft(W)
    out = torch.fft.ifft(torch.einsum('ijk,jrk->irk',Xf,Wf))
    return out.real


class TEncoderlayer(nn.Module):

    def __init__(self, dim_in, dim_out, num_channels, num_nodes, bias=True): 
        super(TEncoderlayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.FloatTensor(dim_in, dim_out, num_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(num_nodes, dim_out, num_channels))    
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, adj):     # features和adj：(num_channels, num_nodes, dim)
        support = fft_product(features, self.weight)
        if adj.is_sparse:
            adj = adj.to_dense()
        output = fft_product(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class TensorLinear(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, dim_2, bias=True):    
        super(TensorLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_nodes = num_nodes
        self.weight = nn.Parameter(torch.FloatTensor(dim_in, dim_out))    # 第三维度上的维度投影
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(num_nodes, dim_2, dim_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.einsum('ijk,kr->ijr', input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):                          # init
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
