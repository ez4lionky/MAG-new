import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules import Module
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops,to_dense_batch,sparse_to_dense,dense_to_sparse
from torch_scatter import scatter_add, scatter_mul
from torch.nn import Sequential, ReLU, Linear
import math
from torch_geometric.transforms import ToDense
import torch.sparse

def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class DiffPoolSparse(torch.nn.Module):
    def __init__(self,batch_size,max_nodes_num,clusters_num):
        super().__init__()
        self.S=torch.nn.Parameter(torch.Tensor(batch_size,max_nodes_num,clusters_num))
        self.max_nodes_num=max_nodes_num
        self.batch_size=batch_size
        self.reset_parameters()

    def forward(self, x,edge_index,batch):
        X_batch=torch.zeros(self.batch_size,self.max_nodes_num,x.size(1),device='cuda')
        X_p,_=to_dense_batch(x,batch)
        real_batch_size=X_p.size(0)
        real_max_nodes_num=X_p.size(1)
        X_batch[:real_batch_size,:real_max_nodes_num,:]=X_p
        A_batch=torch.zeros(self.batch_size,self.max_nodes_num,self.max_nodes_num,device='cuda')
        A_p=torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.size(1),device='cuda'), torch.Size([x.size(0), x.size(0)]))
        A_p=A_p.to_dense()

        for i in range(real_batch_size):
            idx=torch.eq(batch,i)
            a=A_p[idx,:]
            A_batch[i,:sum(idx),:sum(idx)]=a[:,idx]
        X,A,_=dense_diff_pool(X_batch,A_batch,self.S)

        X_l=[]
        batch_coarse = []
        for i in range(real_batch_size):
            nm=self.S.size(-1)
            X_l.append(X[i,:nm,:])
            batch_coarse.append(i * torch.ones(nm,dtype=torch.long))
        batch_coarse = torch.cat(batch_coarse, -1).to('cuda')
        x_coarse=torch.cat(X_l,0)
        ei_l=[]
        for i in range(real_batch_size):
            nm = self.S.size(-1)
            A[i, torch.arange(nm), torch.arange(nm)] = 0
            a=A[i, :nm, :nm]
            z = torch.where(a > 0, torch.ones(nm, nm,device='cuda'), torch.zeros(nm, nm,device='cuda'))
            z = dense_to_sparse(z)[0] + nm * (i)
            #z=z.to('cuda')
            ei_l.append(z)
        edge_index_coarse=torch.cat(ei_l,-1)

        return x_coarse,edge_index_coarse,batch_coarse
    def reset_parameters(self):
        glorot(self.S)


#x=torch.tensor([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9],[10,10,10]])
#edge_index=torch.tensor([[0,0,0,0,1,2,3,4,5,5,6,7,8,8,9,10],[1,2,3,4,0,0,0,0,6,7,5,5,9,10,8,8]],dtype=torch.long)
#batch=torch.tensor([0,0,0,0,0,1,1,1,2,2,2],dtype=torch.long)
#batch_size=4
#max_nodes_num=6
#clusters_num=2
#pool=DiffPoolSparse(batch_size,max_nodes_num,clusters_num)
#x,e,b=pool(x,edge_index,batch)
#print(x)
#print(e)
#print(b)
