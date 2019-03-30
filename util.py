import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt

def plot_loss_and_acc(epoch, train_losses, test_losses, train_accs, test_accs, fpath='../Graphs/cv_fig'):
    plt.figure()
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
    ax2.set_title('D-GAT_SUM Acc - Epoch curve')
    plt.legend(handles=[l3, l4], labels=['Train_accs', 'Test_accs'], loc='best')
    plt.savefig(fpath)


def make_cv(path, i, train_index, test_index):
    # print(path)
    fpath = path + '/cv'
    if not osp.exists(fpath):
        os.makedirs(fpath)
    fpath = fpath + '/cv_fold_{}'.format(i)
    # print(fpath)
    np.savetxt(fpath + '_train_index', train_index.reshape(1, -1), '%d')
    np.savetxt(fpath + '_test_index', test_index.reshape(1, -1), '%d')

def read_cv(path, i):
    fpath = path + '/cv/cv_fold_{}'.format(i)
    train_index = list(np.loadtxt(fpath + '_train_index', dtype='int32'))
    test_index = list(np.loadtxt(fpath + '_test_index', dtype='int32'))

    return  train_index, test_index