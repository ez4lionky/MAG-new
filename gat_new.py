import os.path as osp
import numpy as np
from sklearn.model_selection import KFold
from util import plot_loss_and_acc

import torch
from torch.nn import BatchNorm1d
from torch.nn.modules import Module
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
from attention_conv import GATConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'NCI1')
dataset = TUDataset(path, name='NCI1').shuffle()
data_index = range(len(dataset))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_components = dataset.num_classes

node_features = []
graph_features = []
degree = []

def append_node_features(module, input, output):
    del node_features[:-1]
    node_features.append(output.tolist())

def append_graph_features(module, input, output):
    del graph_features[:-1]
    graph_features.append(input[0].tolist())


def append_degree(module, input, output):
    del degree[:-1]
    degree.append(output.tolist())

class HookDegree(Module):
    def __init__(self, heads):
        super(HookDegree, self).__init__()
        self.heads = heads

    def forward(self, edge_index, x):
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=x.dtype, device=x.device).view(-1)
        loop_weight = torch.full(
                        (x.size(0),),
                        1,
                        dtype=x.dtype,
                        device=x.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        edge_index = add_self_loops(edge_index)
        deg = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=x.size(0))
        return deg


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hook = HookDegree(8)
        self.hook.register_forward_hook(append_degree)
        self.conv1 = GATConv(128*1, 1, heads=128, dropout=0.2, concat=True)
        self.bn1 = BatchNorm1d(128 * 1)
        self.conv2 = GATConv(128 * 1, 1, heads=128, dropout=0.2, concat=True)
        self.bn2 = BatchNorm1d(128 * 1)
        self.conv3 = GATConv(128* 1, 1, heads=128, dropout=0.2, concat=True)
        self.bn3 = BatchNorm1d(128 * 1)
        self.conv4 = GATConv(1 * 128, 1, heads=128, dropout=0.2,concat=True)
        self.bn4 = BatchNorm1d(128 * 1)
        self.conv5 = GATConv(1 * 128, 1, heads=128, dropout=0.2,concat=True)
        self.bn5 = BatchNorm1d(128 * 1)
        self.bn6 = BatchNorm1d(128 * 1)
        self.fc0=Linear(dataset.num_features,128)
        self.concat=torch.cat

        self.fc1 = Linear(128, 64)
        self.fc1.register_forward_hook(append_graph_features)
        self.fc2 = Linear(64, dataset.num_classes)
        #self.fc3 = Linear(64, dataset.num_features)
        #self.fc3.register_forward_hook(append_node_features)

    def forward(self, x, edge_index, batch):
        x = self.fc0(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        z = x
        x = F.elu(self.conv1(x, edge_index))
        x = z+x

        x = self.bn2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        z = x
        x = F.elu(self.conv2(x, edge_index))
        x = z+x

        x = self.bn3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        z = x
        x = F.elu(self.conv3(x, edge_index))
        x = z+x

        x = self.bn4(x)
        x = F.dropout(x, p=0.2, training=self.training)
        z = x
        x = F.elu(self.conv4(x, edge_index))
        x = z+x


        x = self.bn5(x)
        x = F.dropout(x, p=0.2, training=self.training)
        # self.hook(edge_index, x)
        z=x
        x = F.elu(self.conv5(x, edge_index))
        x=z+x
        # self.fc3(x)
        x = global_add_pool(x, batch)


        x = self.bn6(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2 ,training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train(model, optimizer, epoch):
    model.train()

    if epoch % 20 == 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.9 * param_group['lr']

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
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(data_index, data_index):
        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_index = list(train_index)
        test_index = list(test_index)
        cv_train_losses, cv_test_losses, cv_train_accs, cv_test_accs = ([] for i in range(4))
        test_dataset = load_data(dataset, test_index)
        train_dataset = load_data(dataset, train_index)
        test_loader = DataLoader(test_dataset, batch_size=512)
        train_loader = DataLoader(train_dataset, batch_size=512)
        for epoch in range(1, 200):
            train_loss = train(model, optimizer, epoch)
            train_acc, _ = test(model, train_loader)
            test_acc, test_loss = test(model, test_loader)
            print('Epoch: {:03d}, Train Loss: {:.7f}, Test Loss: {:.7f}, '
                  'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss, test_loss,
                                                               train_acc, test_acc))
            cv_train_losses.append(train_loss)
            cv_test_losses.append(test_loss)
            cv_train_accs.append(train_acc)
            cv_test_accs.append(test_acc)

        train_losses.append(cv_train_losses)
        test_losses.append(cv_test_losses)
        train_accs.append(cv_train_accs)
        test_accs.append(cv_test_accs)

    train_losses, test_losses = np.array(train_losses), np.array(test_losses)
    train_losses = np.mean(train_losses, axis=0)
    test_losses = np.mean(test_losses, axis=0)
    train_accs = np.mean(train_accs, axis=0)
    test_accs = np.mean(test_accs, axis=0)

    plot_loss_and_acc(epoch, train_losses, test_losses, train_accs, test_accs)