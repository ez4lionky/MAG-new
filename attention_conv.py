import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules import Module
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add, scatter_mul
from torch.nn import Sequential, ReLU, Linear
import math


alpha_entropy = []
def append_alpha(module, input, output):
    del alpha_entropy[:-1]
    alpha_entropy.append(output.tolist())

def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class HookEntropyLayer(Module):
    def __init__(self):
        super(HookEntropyLayer, self).__init__()

    def forward(self, input, edge_index):
        log = torch.log(input)
        input = torch.mul(log, input)
        entropy = -scatter_add(input, edge_index, dim=0)
        return entropy


class GATConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.hook = HookEntropyLayer()
        self.hook.register_forward_hook(append_alpha)

        self.weight1 = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        if concat:
            self.weight2 = Parameter(
                torch.Tensor(heads * out_channels, heads * out_channels))
        else:
            self.weight2 = Parameter(
                torch.Tensor(out_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        """"""
        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        x = torch.mm(x, self.weight1).view(-1, self.heads, self.out_channels)

        x = self.propagate('add', edge_index, x=x, num_nodes=x.size(0))
        # x = torch.mm(x, self.weight2)
        return x

    def message(self, x_i, x_j, edge_index, num_nodes):
        # Compute attention coefficients.
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.relu(alpha)
        # alpha = torch.exp(-F.leaky_relu(alpha, self.negative_slope))
        # alpha = alpha / scatter_add(alpha, edge_index[0], dim=0, dim_size=num_nodes)[edge_index[0]]
        # self.hook(alpha, edge_index[0])

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        return x_j * alpha.view(-1, self.heads, 1)


    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
