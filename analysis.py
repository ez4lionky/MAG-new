import numpy as np
from attention_conv import alpha_entropy
from gat import Net, dataset, node_features, graph_features, n_components, degree
from util import plot_loss_and_acc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
train_dataset = dataset[n:]
test_loader = DataLoader(test_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
input = {'x': [], 'y': []}

def train(model, optimizer, epoch):
    model.train()

    if epoch == 51 and epoch < 400:
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
        input['x'] = data.x
        input['y'] = data.y
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset), loss_all / len(test_dataset)

train_losses, test_losses, train_accs, test_accs = [], [], [], []
print('Model training...')
# fig=plt.figure()
# plt.ion()
# xs, ys_train_loss, ys_test_loss, ys_train_acc, ys_test_acc = ([0, 0] for _ in range(5))
# ax1, ax2 = plt.subplot(211), plt.subplot(212)
for epoch in range(0, 100):
    if (epoch + 1) % 50==0:
        print('50 epochs done!')
    train_loss = train(model, optimizer, epoch)     # hook append n (train_batches) times
    train_acc, _ = test(model, train_loader)        # hook append n (train_batches) times
    test_acc, test_loss = test(model, test_loader)  # hook append n (test_batches) times

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    # xs[0], ys_train_loss[0], ys_test_loss[0], ys_train_acc[0], ys_test_acc[0] = \
    # xs[1], ys_train_loss[1], ys_test_loss[1], ys_train_acc[1], ys_test_acc[1]
    # xs[1], ys_train_loss[1], ys_test_loss[1], ys_train_acc[1], ys_test_acc[1] = \
    # epoch, train_loss, test_loss, train_acc, test_acc
    # ax1.plot(xs, ys_train_loss, 'b', label='train_loss')
    # ax1.plot(xs, ys_test_loss, 'r', label='test_loss')
    # ax2.plot(xs, ys_train_acc, 'b', label='train_acc')
    # ax2.plot(xs, ys_test_acc, 'r', label='test_acc')
    # plt.pause(0.01)

plot_loss_and_acc(epoch + 1, train_losses, test_losses, train_accs, test_accs)
print('Last epoch, Train Loss: {:.7f}, Test Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(train_loss, test_loss,
                                                       train_acc, test_acc))

node_features = np.array(node_features, dtype='float32')
graph_features = np.array(graph_features, dtype='float32')
x = np.array(input['x'][-1].cpu(), dtype='float32')
y = np.array(input['y'][-1].cpu())
tsne_in = np.concatenate([node_features, x])

ax1 = plt.subplot(1, 2, 1)
tsne = TSNE(n_components=n_components, init='random', random_state=0, perplexity=5)
embed = tsne.fit_transform(tsne_in)
mid = int(len(embed)/2)

ax1.scatter(embed[:mid, 0], embed[:mid, 1], c='r', label='Extracted features')
ax1.scatter(embed[mid:, 0], embed[mid:, 1], c='b', label='Original features')
ax1.legend(loc="upper left")

ax2 = plt.subplot(1, 2, 2)
tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=5)
embed = tsne.fit_transform(graph_features)
ax2.scatter(embed[y==0, 0], embed[y==0, 1], c='r', label='Class 0')
ax2.scatter(embed[y==1, 0], embed[y==1, 1], c='b', label='Class 1')
ax2.legend(loc="upper left")
plt.savefig('../Graph/tsne')

plt.figure(figsize=(10, 10))

alpha_entropy = np.array(alpha_entropy[-1])   # 得到的是加上自环后的attention weight, alpha_weight[-1]为最后一个epoch的最后一层GAT的结果 num = e + n
print(alpha_entropy.shape)
degree = np.array(degree[-1])
print('np.min(alpha_entropy', np.min(alpha_entropy))
print('np.max(alpha_entropy)', np.max(alpha_entropy))
print('np.min(degree)', np.min(degree))
print('np.max(degree)', np.max(degree))

sns.set()
colors = sns.color_palette('hls', 8)

plt.cla()
plt.figure(figsize=(10, 10))
plt.xticks(np.linspace(np.min(degree), np.max(degree), np.max(degree) - np.min(degree) + 1))
plt.yticks(np.linspace(np.min(alpha_entropy), np.max(alpha_entropy), 20))

for i in range(8):
    plt.scatter(degree, alpha_entropy[:, i], s=15, color=colors[i], label='Head {}'.format(i))

plt.legend()
plt.show()
# plt.savefig('../Graph/attention entropy')
