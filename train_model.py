from extract_feature import get_symmetries, get_feature_img
import torch
import numpy as np
from tqdm import tqdm
from cnn import *
import os
from torch.utils.data import TensorDataset, DataLoader
import pdb

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def expand_dataset(X, y):
    if len(X) != len(y):
        print(len(X), len(y))
        raise NameError('Problem with dataset')
    newX = []
    newY = []
    for i in tqdm(range(len(X))):
        syms = get_symmetries(X[i])
        newX += syms
        newY += [y[i] for _ in range(len(syms))]
    return np.array(newX), np.array(newY)

def train_one_epoch(num_epoch, model, dataloader, loss_fn, optimizer):
    model.train()
    model.to(device)
    running_loss = []
    ttp = 0
    tfp = 0
    ttn = 0
    tfn = 0

    pbar = tqdm(dataloader)
    for data in pbar:
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs.to(device))

        loss = loss_fn(outputs.cpu(), labels)
        loss.backward()

        optimizer.step()
        # acc = accuracy(outputs, labels)

        running_loss.append(loss.item())
        tp, fp, tn, fn = metric_infos(outputs, labels)
        ttp += tp
        tfp += fp
        ttn += tn
        tfn += fn
        acc1, acc2 = metrics_2_acc(tp, fp, tn, fn)
        pbar.set_description('num_epoch: {}, loss: {:.2f}, bact_acc: {:.2f}, res_acc: {:.2f}'.format(num_epoch, loss.item(), acc1, acc2))

    # print('numiters: {}, loss: {}, accuracy: {}'.format(num_epochs, loss.item(), acc))
    acc1, acc2 = metrics_2_acc(ttp, tfp, ttn, tfn)
    return [np.mean(running_loss)], [acc1], [acc2]

def validate(val_X, val_Y, model, loss_fn):
    model.eval().cpu()
    outputs = model(val_X.cpu())

    loss = loss_fn(outputs, val_Y.cpu())
    acc = accuracy(outputs, val_Y.cpu())
    print('validation loss: {:.2f}'.format(loss.item()), end = ' ')
    print('validation accuracy: {}'.format(acc))
    return loss.item(), acc[0], acc[1]

def accuracy(outputs, labels):
    tp, fp, tn, fn = metric_infos(outputs, labels)
    acc1, acc2 = metrics_2_acc(tp, fp, tn, fn)
    return np.array([acc1, acc2])

def metrics_2_acc(tp, fp, tn, fn):
    acc1 = 0 if (tp + fn) == 0 else tp / (tp + fn)
    acc2 = 0 if (tn + fp) == 0 else tn / (tn + fp)
    return acc1, acc2

# TP, FP, TN, FN
def metric_infos(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    true_positive = int(((preds == 0) & (preds == labels)).sum())
    false_positive = int(((preds == 0) & (preds != labels)).sum())
    true_negative = int(((preds == 1) & (preds == labels)).sum())
    false_negative = int(((preds == 1) & (preds != labels)).sum())
    return true_positive, false_positive, true_negative, false_negative


def train_loop(num_epochs, model_path, log_path, model_type = "lenet", load_path = None, lr=1e-5):
    ## load_path vairable used be a tuple of (log_path, model_path)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    if not os.path.isdir(log_path):
        os.mkdir(log_path)


    # Loading data
    valX = np.load("extracted_data/testX.npy", allow_pickle=True)
    valX = get_feature_img(valX[:, 0], valX[:, 1]) # Normalize image with background mean
    valY = np.load('extracted_data/testY.npy', allow_pickle=True)
    # print('expanding validation_set')
    # valX, valY = expand_dataset(valX, valY)
    # print('expanded')
    valX = torch.Tensor(reshape_data(valX))
    valY = torch.Tensor(valY).long()

    trainX = np.load("extracted_data/trainX.npy", allow_pickle=True)
    trainX = get_feature_img(trainX[:, 0], trainX[:, 1])
    trainY = np.load('extracted_data/trainY.npy', allow_pickle=True)
    print('expanding trainset')
    # trainX, trainY = expand_dataset(trainX, trainY)
    print('expanded')
    trainX = torch.Tensor(reshape_data(trainX))
    trainY = torch.Tensor(trainY).long()


    trainset = TensorDataset(trainX, trainY)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

    input_shape = valX[0][0].shape
    np.save(model_path + '/input_shape', input_shape)

    # initiation succeeeded

    loss_fn = nn.CrossEntropyLoss()
    if model_type == "lenet":
        model = LeNet5((3, *input_shape), 2)
    elif model_type == "resnet":
        model = ResNetMNIST(2)
    else:
        print('unknown model type, I will just use lenet')
        model = LeNet5((3, *input_shape), 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if load_path is not None:
        log_path = load_path[0]
        model_path = load_path[1]
        losses = list(np.load(log_path + '/loss_hist.npy'))
        accs1 = list(np.load(log_path + '/acc1_hist.npy'))
        accs2 = list(np.load(log_path + '/acc2_hist.npy'))
        val_losses = list(np.load(log_path + '/val_loss.npy'))
        val_acc1 = list(np.load(log_path + '/val_acc1.npy'))
        val_acc2 = list(np.load(log_path + '/val_acc2.npy'))
        num = len(val_acc2)
        model.load_state_dict(torch.load(model_path + "/bact_model_{}".format(num)))
        # pdb.set_trace()
    else:
        losses = []
        accs1 = []
        accs2 = []

        val_losses = []
        val_accs1 = []
        val_accs2 = []
        num = 0
    for i in range(num + 1, num + num_epochs + 1):
        loss, acc1, acc2 = train_one_epoch(i, model, train_loader, loss_fn, optimizer)
        val_loss, val_acc1, val_acc2 = validate(valX, valY, model, loss_fn)

        losses += loss
        accs1 += acc1
        accs2 += acc2

        val_losses.append(val_loss)
        val_accs1.append(val_acc1)
        val_accs2.append(val_acc2)

        torch.save(model.state_dict(), model_path + '/bact_model_{}'.format(i))
        np.save(log_path + '/loss_hist', losses)
        np.save(log_path + '/acc1_hist', accs1)
        np.save(log_path + '/acc2_hist', accs2)
        np.save(log_path + '/val_loss', val_losses)
        np.save(log_path + '/val_acc1', val_accs1)
        np.save(log_path + '/val_acc2', val_accs2)

    
if __name__ == "__main__":
    # train_loop(150, 'models', 'logs', load_path=['logs', 'models'])
    train_loop(150, 'grad_model', 'grad_log', load_path=None)