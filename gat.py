import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool
from attention_conv import GATConv
from torch_geometric.nn.inits import reset
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops, add_self_loops

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MUTAG')
dataset = TUDataset(path, name='MUTAG').shuffle()
data_index = range(len(dataset))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6, concat=True)
        # self.conv2 = GATConv(32 * 1, 1, heads=32, dropout=0.6, concat=True)
        # self.conv3 = GATConv(32 * 1, 1, heads=32, dropout=0.6, concat=True)
        self.conv4 = GATConv(8 * 8, 8, heads=8, dropout=0.6)

        self.fc1 = Linear(64, 32)
        self.fc2 = Linear(32, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        # x = F.elu(self.conv2(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = F.elu(self.conv3(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv4(x, edge_index)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)



def train(model, optimizer, epoch):
    model.train()

    if epoch == 51 and epoch<400:
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


train_losses, test_losses, train_accs, test_accs = [], [], [], []
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(data_index, data_index):
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    train_index = list(train_index)
    test_index = list(test_index)
    cv_train_losses, cv_test_losses, cv_train_accs, cv_test_accs = ([] for i in range(4))
    test_dataset = load_data(dataset, test_index)
    train_dataset = load_data(dataset, train_index)
    test_loader = DataLoader(test_dataset, batch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=128)
    for epoch in range(1, 801):
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

ax1 = plt.subplot(211)
l1, = plt.plot(range(epoch), train_losses, 'b', label='train_losses')
l2, = plt.plot(range(epoch), test_losses, 'r', label='test_losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout(h_pad=3)
plt.legend(handles=[l1, l2], labels=['Train_losses', 'Test_losses'], loc='best')
ax1.set_title('D-GAT_SUM Loss - Epoch curve')

ax2 = plt.subplot(212)
l3, = plt.plot(range(epoch), train_accs, 'b', label='train_accs')
l4, = plt.plot(range(epoch), test_accs, 'r', label='test_accs')
plt.text(epoch, train_accs[-1], '{:.4f}'.format(train_accs[-1]), ha='center', va='bottom', fontsize=10)
plt.text(epoch, test_accs[-1], '{:.4f}'.format(test_accs[-1]), ha='center', va='top', fontsize=10)
plt.xlabel('Epochs')
plt.ylabel('Accs')
plt.tight_layout(h_pad=3)
ax2.set_title('GIN Acc - Epoch curve')
plt.legend(handles=[l3, l4], labels=['Train_accs', 'Test_accs'], loc='best')
plt.show()
# plt.savefig('Graphs/')
