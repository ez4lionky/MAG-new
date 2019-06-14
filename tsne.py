import time
import argparse
from util import *
from sklearn.manifold import TSNE

import torch
from torch.nn import BatchNorm1d
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from EdgeConv import EdgeConv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser(description='Edge convolutional network for graph classification')
parser.add_argument('--dataset_name', type=str, default='MUTAG',
                    help='Dataset name (default: MUTAG)')
parser.add_argument('--batch_size', type=int, default=47,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=201,
                    help='number of epochs to train (default: 350)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=100,
                    help='random seed for splitting the dataset into 10 (default: 0)')
parser.add_argument('--num_blocks', type=int, default=3,
                    help='number of layers INCLUDING the input one (default: 3)')
parser.add_argument('--edge_filters_dim', type=int, default=8,
                    help='dimension of edge filter (default: 8)')
parser.add_argument('--block_in_dim', type=str, default='16-16-16',
                    help='the transformed input dimension and number of pre-layer\'s subgraph edge filter (default: \'8-32-32\')')
parser.add_argument('--block_out_dim', type=str, default='16-16-16',
                    help='number of current layer\'s subgraph edge filter (default: \'32-32-32\')')
parser.add_argument('--mlp_dim', type=int, default=8,
                    help='number of hidden units (default: 64)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='number of hidden units (default: 1e-4)')
parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate(default: 0.2)')
args = parser.parse_args()


max_degree = 1000

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset_name)
result_path = osp.join(osp.dirname(osp.realpath(__file__)),  '..', 'Results', args.dataset_name, 'tmp.txt')
dataset = TUDataset(
    path,
    name=args.dataset_name,
    # transform=T.OneHotDegree(max_degree),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_components = dataset.num_classes

blocks = args.num_blocks
block_in_channels = [int(_) for _ in args.block_in_dim.split('-')]
block_out_channels = [int(_) for _ in args.block_out_dim.split('-')]
edge_filters_num_list = [args.edge_filters_dim for _ in range(blocks)]
learning_rate = args.lr
dropout_rate = args.dropout
batch_size = args.batch_size
print(args)

graph_features = []
label = []
n = len(dataset) // batch_size
hook = 2

def append_graph_features(module, input, output):
    # graph_features.append(input[0].tolist())
    graph_features.append(output.tolist())

class MyRelu(torch.nn.Module):
    def __init__(self):

        super().__init__()
    def forward(self, x):
        return F.relu(x)

class MyLogSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.log_softmax(x, dim=-1)

class MyEdgeConvBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,edge_filters_num,dropout=dropout_rate):
        super().__init__()
        self.bn = BatchNorm1d(in_channels)
        self.ecn = EdgeConv(in_channels, out_channels, edge_filters_num,dropout)

    def forward(self, x,edge_index):
        x = self.bn(x)
        x = self.ecn(x,edge_index)
        x = F.relu(x)
        return x

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc0 = Linear(dataset.num_features, block_in_channels[0])

        self.blocks0 = torch.nn.ModuleDict()
        for i in range(blocks):
            self.blocks0['block' + str(i)] = MyEdgeConvBlock(block_in_channels[i],block_out_channels[i],edge_filters_num_list[i],dropout_rate)
            # if i == hook:
            #     self.hook_layer = Linear(block_in_channels[i], block_out_channels[i])
            #     self.hook_layer.register_forward_hook(append_graph_features)

        self.fc1 = Linear(sum(block_out_channels), args.mlp_dim)
        # self.fc1 = Linear(block_out_channels[-1], args.mlp_dim)
        self.fc2 = Linear(args.mlp_dim, dataset.num_classes)
        self.fc2.register_forward_hook(append_graph_features)

    def forward(self, x, edge_index, batch):
        x = self.fc0(x)
        x = F.dropout(x,dropout_rate)
        xlist = []

        for i in range(blocks):
            x = self.blocks0['block' + str(i)](x, edge_index)
            x = F.dropout(x, dropout_rate)
            xlist.append(x)
            # if i== hook:
            #     tmp = global_mean_pool(x, batch)
            #     tmp = self.hook_layer(tmp)

        x = torch.cat(xlist,-1)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, dropout_rate)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=-1)
        return x


def train(model, optimizer, epoch):
    model.train()

    if epoch % 50 == 0:
       for param_group in optimizer.param_groups:
           param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        label.append(data.y.detach().cpu().numpy())
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(dataset)


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
    return correct / len(loader.dataset), loss_all / len(dataset)


def load_data(data, index):
    final = []
    for i in index:
        i = int(i)
        final.append(data[i])
    return final


if __name__ == '__main__':
    np.set_printoptions(threshold=10000)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(dataset, batch_size=batch_size)
    for epoch in range(1, args.epochs):
        del label[:]
        start = time.time()
        train_loss = train(model, optimizer, epoch)
        # train_acc, _ = test(model, train_loader)
        end = time.time()
        # line = ('Epoch: {:03d}, Train Loss: {:.7f}, '
        # 'Train Acc: {:.7f}, Time: {:.2f}'.format(epoch, train_loss, train_acc, end - start))
        line = ('Epoch: {:03d}, Train Loss: {:.7f}, '
        'Time: {:.2f}'.format(epoch, train_loss, end - start))
        print(line)

    graph_features = np.array(graph_features, dtype='float32')
    last_epoch_gf = graph_features[-n:]
    last_epoch_gf = np.concatenate(last_epoch_gf, axis=0)

    # tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=5)
    # embed = np.array(tsne.fit_transform(last_epoch_gf))
    # label = np.array(label)
    # x_min, x_max = np.min(embed, 0), np.max(embed, 0)
    # embed = embed / (x_max - x_min)

    label = np.concatenate(label, axis=0)
    tsne = TSNE(n_components=3, perplexity=20)
    embed = np.array(tsne.fit_transform(last_epoch_gf))
    x_min, x_max = np.min(embed, 0), np.max(embed, 0)
    embed = embed / (x_max - x_min)

    # 创建显示的figure
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(embed[label == 0, 0], embed[label == 0, 1], embed[label == 0, 2], c=plt.cm.Set2(label[label == 0]), label="Category 1")
    # ax.scatter(embed[label == 1, 0], embed[label == 1, 1], embed[label == 1, 2], c=plt.cm.Set2(label[label == 1]), label="Category 2")

    ax.scatter(embed[label == 0, 0], embed[label == 0, 1], embed[label == 0, 2], c=plt.cm.Set1((label[label == 0] + 1) / 10), label="Category 1")
    ax.scatter(embed[label == 2, 0], embed[label == 2, 1], embed[label == 2, 2], c=plt.cm.Set1((label[label == 2]) / 10), label="Category 2")

    plt.legend()
    # plt.savefig("PPGCN-"+ args.dataset_name + "-layer" + str(hook))
    plt.savefig("PPGCN-MUTAG-concat.png")
    plt.show()