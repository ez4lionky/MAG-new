import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules import Module
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add, scatter_mul
from torch.nn import Sequential, ReLU, Linear
import math
from torch.nn import BatchNorm1d

def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

#out_channels=20
#in_channels=10
#edge_filters_num=16
#subgraph_filters_num=32

class MyLinear(torch.nn.Module):
    def __init__(self,subgraph_filters_num,edge_filters_num):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.Tensor(subgraph_filters_num,edge_filters_num,1) ##(subgraph_filters_num,edge_filters_num,1)
        )
        self.bias = torch.nn.Parameter(
            torch.Tensor(subgraph_filters_num, 1)
        )
        self.reset_parameters()
    def forward(self, x):
        x = F.relu(torch.matmul(x,self.weights)+self.bias)
        return x
    def reset_parameters(self):
        glorot(self.weights)
        zeros(self.bias)


class EdgeConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_filters_num=16,
                 dropout=0):
        super(EdgeConv, self).__init__(aggr='add')
        self.edge_filters_num=edge_filters_num
        self.subgraph_filters_num=out_channels #subgraph_filters_num
        self.in_channels=in_channels
        self.dropout=dropout
        ##############################################
        self.subgraph_edges_filters_weights = torch.nn.Parameter(
            torch.Tensor(self.in_channels * 2, self.edge_filters_num * self.subgraph_filters_num)
        )
        ##(node_nums*2,edge_filters_num*subgraph_filters_num)
        #print(self.subgraph_edges_filters_weights.size())
        self.subgraph_filters_bias = torch.nn.Parameter(
            torch.Tensor(1, self.edge_filters_num * self.subgraph_filters_num)
        ) ##(1,edge_filters_num*subgraph_filters_num)
        #print(self.subgraph_filters_bias.size())

        self.affine_weights0=torch.nn.Parameter(
            torch.Tensor(self.subgraph_filters_num, self.edge_filters_num, self.edge_filters_num) # (subgraph_filters_num, edge_filters_num, edge_filters_num)
        )

        self.affine_bias0=torch.nn.Parameter(
            torch.Tensor(self.subgraph_filters_num, 1, self.edge_filters_num)
        )

        self.affine_weights1 = torch.nn.Parameter(
            torch.Tensor(self.subgraph_filters_num, self.edge_filters_num, 1)
            ##(subgraph_filters_num,edge_filters_num,1)
        )

        self.affine_bias1=torch.nn.Parameter(
            torch.Tensor(self.subgraph_filters_num, 1, 1)
        )
        self.bn=BatchNorm1d(self.subgraph_filters_num*self.edge_filters_num)
        #print(self.affine_weights.size())
        ###############################################
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.subgraph_edges_filters_weights)
        glorot(self.affine_weights0)
        glorot(self.affine_weights1)
        zeros(self.affine_bias1)
        zeros(self.affine_bias0)
        zeros(self.subgraph_filters_bias)

    def forward(self, x, edge_index):
        # x has shape [nodes_num, in_channels]
        # edge_index has shape [2, edges_num]
        # Step 3-5: Start propagating messages.
        return self.propagate('add',edge_index, x=x, nodes_num=x.size(0))

    def message(self, x_i,x_j, edge_index, nodes_num):
        # x_j has shape [edges_num, out_channels]

        # Step 3: Normalize node features.
        # row, col = edge_index
        e_ij =  torch.cat((x_i, x_j), dim=-1)  # [nodes_num, in_channels * 2]
        out = F.relu(torch.mm(e_ij, self.subgraph_edges_filters_weights) + self.subgraph_filters_bias)
        # (edges_num, subgraph_filters_num * edge_filters_num)

        #if self.training and self.dropout > 0:
        #    out = F.dropout(out, p=self.dropout, training=self.training)

        return out

    def update(self, aggr_out):
        # aggr_out has shape [nodes_num, out_channels]
        aggr_out = self.bn(aggr_out)
        aggr_out = aggr_out.view(-1, self.subgraph_filters_num, 1, self.edge_filters_num)
        # [nodes_num,subgraph_filters_num,1,edge_filters_num]
        # aggr_out= F.relu(torch.matmul(aggr_out, self.affine_weights0)+self.affine_bias0)
        if self.training and self.dropout > 0:
            aggr_out = F.dropout(aggr_out, p=self.dropout, training=self.training)
        # [nodes_num,subgraph_filters_num,1,edge_filters_num]
        aggr_out = F.relu(torch.matmul(aggr_out, self.affine_weights0) + self.affine_bias0)

        aggr_out = F.relu(torch.matmul(aggr_out, self.affine_weights1) + self.affine_bias1)
        # [nodes_num,subgraph_filters_num,1,1]
        aggr_out.squeeze_()  # (nodes_num,subgraph_filters_num)
        return aggr_out
    def __repr__(self):
        return '{}({}, {}, edge_filters_num={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.subgraph_filters_num,
                                             self.edge_filters_num
                                             )

#layer=EdgeConv(3,5)
#print(layer)
#x=torch.tensor([[0.,0.,0.],[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]])
#edge_index=torch.tensor([[0, 0, 1, 1 ,1, 2, 2, 3, 3 ,3],[1 ,3, 0, 2, 3 ,1, 3, 0 ,1 ,2]],dtype=torch.long)
#layer.eval()
#z=layer(x,edge_index)