import time
import numpy as np
import os.path as osp
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import EdgeConv
from torch_geometric.nn import global_mean_pool

max_degree = 10000
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'IMDB-BINARY')
dataset = TUDataset(
    path,
    name='IMDB-BINARY',
    transform=T.OneHotDegree(max_degree)
)
label = dataset.data.y
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(data, index):
    final = []
    for i in index:
        i = int(i)
        final.append(data[i])
    return final


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.eg1 = EdgeConv(
            Seq(Lin(dataset.num_features * 2, 64), ReLU(), Lin(64, 32), ReLU()), 'add')

        self.eg2 = EdgeConv(
            Seq(Lin(64, 64), ReLU(), Lin(64, 32), ReLU()), 'add')

        self.eg3 = EdgeConv(
            Seq(Lin(64, 64), ReLU(), Lin(64, 32), ReLU()), 'add')

        self.eg4 = EdgeConv(
            Seq(Lin(64, 64), ReLU(), Lin(64, 32), ReLU()), 'add')

        self.eg5 = EdgeConv(Seq(Lin(64, 64), ReLU(), Lin(64, 64), ReLU()), 'add')

        self.lin1 = Lin(64, 32)
        self.lin2 = Lin(32, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.eg1(x, edge_index)
        x = self.eg2(x, edge_index)
        x = self.eg3(x, edge_index)
        x = self.eg4(x, edge_index)
        x = self.eg5(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=-1)


def train(model,  optimizer, epoch):
    model.train()

    if epoch % 50 == 0:
       for param_group in optimizer.param_groups:
           param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data.x, data.edge_index, data.batch), data.y)
        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)


def test(model, loader):
    model.eval()
    loss_all = 0
    correct = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss_all += loss.item() * data.num_graphs
            pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset), loss_all / len(test_dataset)


train_losses, test_losses, train_accs, test_accs = [], [], [], []
skf = StratifiedKFold(n_splits=10, random_state=100)
i = 0
for train_index, test_index in skf.split(range(len(label)), label):
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    cv_train_losses, cv_test_losses, cv_train_accs, cv_test_accs = ([] for i in range(4))
    test_dataset = load_data(dataset, test_index)
    train_dataset = load_data(dataset, train_index)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    for epoch in range(1, 201):
        start = time.time()
        train_loss = train(model, optimizer, epoch)
        train_acc, _ = test(model, train_loader)
        # train_loss, train_acc = train(model, optimizer, epoch, train_dataset)
        test_acc, test_loss = test(model, test_loader)

        end = time.time()
        line = ('Epoch: {:03d}, Train Loss: {:.7f}, Test Loss: {:.7f}, '
                'Train Acc: {:.7f}, Test Acc: {:.7f}, Time: {:.2f}'.format(epoch, train_loss, test_loss,
                                                                           train_acc, test_acc, end - start))
        if epoch % 2 == 0:
            print(line)
        cv_train_losses.append(train_loss)
        cv_test_losses.append(test_loss)
        cv_train_accs.append(train_acc)
        cv_test_accs.append(test_acc)

    train_losses.append(cv_train_losses)
    test_losses.append(cv_test_losses)
    train_accs.append(cv_train_accs)
    test_accs.append(cv_test_accs)

train_losses = np.mean(train_losses, axis=0)
test_losses = np.mean(test_losses, axis=0)
train_accs = np.mean(train_accs, axis=0)
test_accs = np.mean(test_accs, axis=0)
print('Mean accuracy', test_accs[-1])
