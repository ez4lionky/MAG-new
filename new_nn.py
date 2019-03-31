import time
import os.path as osp
import numpy as np
from sklearn.model_selection import StratifiedKFold
from util import *

import torch
from torch.nn import BatchNorm1d
from torch.nn.modules import Module
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
from EdgeConv import EdgeConv
from pool_layer import DiffPoolSparse

max_degree = 10000

dataset_name = 'MUTAG'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)
result_path = osp.join(osp.dirname(osp.realpath(__file__)),  '..', 'Results', dataset_name, 'tmp.txt')
dataset = TUDataset(
    path,
    name=dataset_name,
    # transform=T.OneHotDegree(max_degree),
)
label = dataset.data.y
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
n_components = dataset.num_classes

blocks = 5
block_in_channels = [32, 32, 32, 32, 32]
block_out_channels = [32, 32, 32, 32, 32]
edge_filters_num_list = [16, 16, 16, 16, 16]
learning_rate = 0.01
dropout_rate = 0.2
batch_size = 128

class MyRelu(torch.nn.Module):
    def __init__(self):

        super().__init__()
    def forward(self, x):
        return F.relu(x)
class MyLogSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.log_softmax(x,dim=-1)

class MyEdgeConvBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,edge_filters_num,dropout=dropout_rate):
        super().__init__()
        self.bn=BatchNorm1d(in_channels)
        self.ecn=EdgeConv(in_channels, out_channels, edge_filters_num,dropout)

    def forward(self, x,edge_index):
        x=self.bn(x)
        x=self.ecn(x,edge_index)
        x=F.relu(x)
        return x

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc0 = Linear(dataset.num_features, block_in_channels[0])

        self.blocks0 = torch.nn.ModuleDict()
        for i in range(blocks):
            self.blocks0['block' + str(i)] = MyEdgeConvBlock(block_in_channels[i],block_out_channels[i],edge_filters_num_list[i],dropout_rate)
        # self.dp0=DiffPoolSparse(batch_size, 100, 10)

        # self.blocks1 = torch.nn.ModuleDict()
        # for i in range(blocks):
        #    self.blocks1['block' + str(i)] = MyEdgeConvBlock(block_in_channels, block_out_channels,
        #                                                    edge_filters_num_list[i], 0)
        # self.dp1 = DiffPoolSparse(batch_size, 10, 5)

        self.fc = torch.nn.Sequential(
            Linear(sum(block_out_channels), 64),
            MyRelu(),
            Linear(64, dataset.num_classes),
            MyLogSoftmax()
        )

    def forward(self, x, edge_index, batch):
        x = self.fc0(x)
        x = F.dropout(x,dropout_rate)
        #z = x
        xlist=[]
        for i in range(blocks):
            x = self.blocks0['block' + str(i)](x,edge_index)
            x = F.dropout(x, dropout_rate)
            xlist.append(x)
        #    x = x + z
        #    z = x
        # x,edge_index,batch=self.dp0(x,edge_index,batch)
        # z=x
        # for i in range(blocks):
        #    x = self.blocks1['block' + str(i)](x,edge_index)
        #    x = F.dropout(x, dropout_rate)
        #    x = x + z
        #    z = x
        #x, edge_index, batch = self.dp1(x, edge_index, batch)
        #z=x

        x = torch.cat(xlist,-1)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, dropout_rate)
        x = self.fc(x)
        return x


def train(model, optimizer, epoch):
    model.train()

    if epoch % 50 == 0 :
       for param_group in optimizer.param_groups:
           param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)


def test(model, loader):
    model.eval()

    loss_all = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset), loss_all / len(test_dataset)


def load_data(data, index):
    final = []
    for i in index:
        i = int(i)
        final.append(data[i])
    return final


if __name__ == '__main__':
    np.set_printoptions(threshold=10000)
    torch.manual_seed(0)
    np.random.seed(0)
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    skf = StratifiedKFold(n_splits=10)
    i = 0
    for train_index, test_index in skf.split(range(len(label)), label):
        # make_cv(path, i, train_index, test_index)
        train_index, test_index = read_cv(path, i)
        i += 1

        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        train_index = list(train_index)
        test_index = list(test_index)
        cv_train_losses, cv_test_losses, cv_train_accs, cv_test_accs = ([] for i in range(4))
        test_dataset = load_data(dataset, test_index)
        train_dataset = load_data(dataset, train_index)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        for epoch in range(1, 201):
            start = time.time()
            train_loss = train(model, optimizer, epoch)
            train_acc, _ = test(model, train_loader)
            test_acc, test_loss = test(model, test_loader)
            end = time.time()
            line = ('Epoch: {:03d}, Train Loss: {:.7f}, Test Loss: {:.7f}, '
            'Train Acc: {:.7f}, Test Acc: {:.7f}, Time: {:.2f}'.format(epoch, train_loss, test_loss,
                                                                       train_acc, test_acc, end - start))
            write_result(line, result_path)
            if epoch % 2 == 0:
              print(line)
            cv_train_losses.append(train_loss)
            cv_test_losses.append(test_loss)
            cv_train_accs.append(train_acc)
            cv_test_accs.append(test_acc)

        # plot_loss_and_acc(epoch, cv_train_losses, cv_test_losses, cv_train_accs, cv_test_accs, fpath='../Graphs/cv_fig')
        train_losses.append(cv_train_losses)
        test_losses.append(cv_test_losses)
        train_accs.append(cv_train_accs)
        test_accs.append(cv_test_accs)


    train_losses = np.mean(train_losses, axis=0)
    test_losses = np.mean(test_losses, axis=0)
    train_accs = np.mean(train_accs, axis=0)
    test_accs = np.mean(test_accs, axis=0)

    # Change filename
    os.rename(result_path, result_path[:-7] + dataset_name + '_acc_{:.5f}.txt'.format(test_accs[-1]))
    # write_result()
    graph_path = dataset_name + '/' + dataset_name + '_acc_{:.5f}.png'.format(test_accs[-1])
    plot_loss_and_acc(epoch, train_losses, test_losses, train_accs, test_accs, graph_path)
