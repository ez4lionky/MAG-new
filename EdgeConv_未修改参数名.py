import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules import Module
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
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
#heads_num=16
#out_channels=32

class EdgeConv(MessagePassing):

    def __init__(self,
                 in_channels,  # 16
                 out_channels, # 16
                 heads_num=16, # 8
                 dropout=0):
        super(EdgeConv, self).__init__(aggr='add')
        self.heads_num = heads_num
        self.out_channels = out_channels # out_channels
        self.in_channels = in_channels
        self.dropout = dropout

        ##############################################

        self.subgraph_edges_filters_weights = torch.nn.Parameter(
            torch.Tensor(self.in_channels * 2, self.heads_num * self.out_channels)
        )


        ##(node_nums*2,heads_num*out_channels)
        #print(self.subgraph_edges_filters_weights.size())
        self.subgraph_filters_bias = torch.nn.Parameter(
            torch.Tensor(1, self.heads_num * self.out_channels)
        )
        ##(1, heads_num * out_channels)
        #print(self.subgraph_filters_bias.size())

        self.affine_weights0 = torch.nn.Parameter(
            torch.Tensor(self.out_channels * self.heads_num, self.out_channels) # (out_channels, heads_num, heads_num)
        )

        self.affine_bias0 = torch.nn.Parameter(
            torch.Tensor(self.out_channels)
        )

        self.affine_weights1 = torch.nn.Parameter(
            torch.Tensor(self.out_channels, self.out_channels)
            ##(out_channels,heads_num,1)
        )

        self.affine_bias1 = torch.nn.Parameter(
            torch.Tensor(self.out_channels)
        )

        self.bn=BatchNorm1d(self.out_channels * self.heads_num)
        self.att1 = Parameter(torch.Tensor(2 * heads_num * out_channels, heads_num))
        self.att2 = Parameter(torch.Tensor(heads_num, 1))

        # self.att = Parameter(torch.Tensor(1, 2 * heads_num * out_channels))
        self.att = Parameter(torch.Tensor(1, heads_num, 2 * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.subgraph_edges_filters_weights)
        glorot(self.affine_weights0)
        glorot(self.affine_weights1)
        zeros(self.affine_bias1)
        zeros(self.affine_bias0)
        glorot(self.att1)
        glorot(self.att2)
        # glorot(self.att)
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
        # (edges_num, out_channels *

        message_i = scatter_add(out, edge_index[0], dim=0, dim_size=out.size(0))
        message_j = out[edge_index[1]]
        message_alpha = torch.cat([message_i, message_j], dim=-1)
        alpha = torch.matmul(message_alpha, self.att1)
        alpha = F.relu(alpha)
        alpha = torch.matmul(alpha, self.att2)
        alpha = torch.sigmoid(alpha)


        # out = out.view(-1, self.heads_num, self.out_channels)
        # out_i = out[edge_index[0]]
        # out_j = out[edge_index[1]]
        # out_ij = torch.cat((out_i, out_j), dim=-1)
        # alpha = (out_ij * self.att).sum(dim=-1)
        # alpha = F.leaky_relu(alpha, 0.2)
        # alpha = softmax(alpha, edge_index[0], nodes_num).view(-1, self.heads_num, 1)

        out = out * alpha
        if self.training and self.dropout > 0:
           out = F.dropout(out, p=self.dropout, training=self.training)

        return out  # .view(-1, self.out_channels * self.heads_num)

    def update(self, aggr_out):
        # aggr_out has shape [nodes_num, out_channels]
        aggr_out = self.bn(aggr_out)
        # aggr_out = aggr_out.view(-1, self.out_channels, 1, self.heads_num)
        # # [nodes_num,out_channels,1,heads_num]
        # # aggr_out= F.relu(torch.matmul(aggr_out, self.affine_weights0)+self.affine_bias0)
        if self.training and self.dropout > 0:
            aggr_out = F.dropout(aggr_out, p=self.dropout, training=self.training)
        # # [nodes_num,out_channels,1,heads_num]
        aggr_out = F.relu(torch.matmul(aggr_out, self.affine_weights0) + self.affine_bias0)
        aggr_out = F.relu(torch.matmul(aggr_out, self.affine_weights1) + self.affine_bias1)
        # # [nodes_num,out_channels,1,1]
        aggr_out.squeeze_()  # (nodes_num,out_channels)
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads_num={})'.format(self.__class__.__name__,
             self.in_channels,
             self.out_channels,
             self.heads_num
         )

#layer=EdgeConv(3,5)
#print(layer)
#x=torch.tensor([[0.,0.,0.],[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]])
#edge_index=torch.tensor([[0, 0, 1, 1 ,1, 2, 2, 3, 3 ,3],[1 ,3, 0, 2, 3 ,1, 3, 0 ,1 ,2]],dtype=torch.long)
#layer.eval()
#z=layer(x,edge_index)