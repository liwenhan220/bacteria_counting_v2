from torch import nn
import torch
import pdb
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LeNet5(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        sim_input = torch.zeros((1, *input_shape))
        sim_out = torch.flatten(self.layer2(self.layer1(sim_input)))
        self.fc = nn.Linear(len(sim_out), 520)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(520, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return nn.Softmax(dim=1)(out)
        # return out

def train_one_epoch(epoch_index, model, train_loader, loss_fn, optimizer):
    model.train()
    model.to(device)
    running_loss = []
    running_acc = []
    last_loss = 0.0
    last_acc = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    pbar = tqdm(train_loader, total=len(train_loader))
    for data in pbar:
        pbar.set_description('epoch: {}, loss: {:.2f}, acc: {:.2f}'.format(epoch_index, last_loss, np.mean(last_acc)))
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs.to(device))

        # Compute the loss and its gradients
        loss = loss_fn(outputs.cpu(), labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss.append(loss.item())
        last_loss = loss.item()
        last_acc = accuracy(outputs, labels)
        running_acc.append(last_acc)

    return np.mean(running_loss), np.mean(running_acc, axis=0)

def validate(val_X, val_Y, model, loss_fn):
    model.eval().cpu()
    outputs = model(val_X.cpu())

    loss = loss_fn(outputs, val_Y.cpu())
    acc = accuracy(outputs, val_Y.cpu())
    print('validation loss: {:.2f}'.format(loss.item()), end = ' ')
    print('validation accuracy: {}'.format(acc))
    return loss.item(), acc

def accuracy(outputs, labels):
    correct_count = np.zeros(len(outputs[0]))
    total_count = np.zeros(len(outputs[0]))
    for i, pred in enumerate(outputs):
        total_count[labels[i]] += 1
        if torch.argmax(pred) == labels[i]:
            correct_count[labels[i]] += 1
    final_acc = np.zeros(len(outputs[0]), dtype = np.float32)
    for i in range(len(total_count)):
        if total_count[i] != 0:
            final_acc[i] = correct_count[i] / total_count[i]
    return final_acc

def reshape_data(data):
    data_len, width, height, channels = data.shape
    return np.array(data.reshape((data_len, channels, width, height)), dtype=np.float32)

def train_loop(num_epochs, model_path, log_path):
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    if not os.path.isdir(log_path):
        os.mkdir(log_path)


    valX = np.load("extracted_data/testX.npy")

    # pdb.set_trace()
    valX = torch.Tensor(reshape_data(valX))
    valY = torch.Tensor(np.load("extracted_data/testY.npy")).long()

    trainX = np.load("extracted_data/trainX.npy")

    trainX = torch.Tensor(reshape_data(trainX))
    trainY = torch.Tensor(np.load("extracted_data/trainY.npy")).long()

    trainset = TensorDataset(trainX, trainY)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

    valset = TensorDataset(valX, valY)
    val_loader = DataLoader(valset)

    input_shape = valX[0][0].shape
    np.save(model_path + '/input_shape', input_shape)


    loss_fn = nn.CrossEntropyLoss()
    model = LeNet5((3, *input_shape), 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    losses = []
    accs = []

    val_losses = []
    val_accs1 = []
    val_accs2 = []

    for i in range(num_epochs):
        loss, acc = train_one_epoch(i, model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = validate(valX, valY, model, loss_fn)

        losses.append(loss)
        accs.append(acc)

        val_losses.append(val_loss)
        val_accs1.append(val_acc[0])
        val_accs2.append(val_acc[1])

        torch.save(model.state_dict(), model_path + '/bact_model_{}'.format(i))
        np.save(log_path + '/loss_hist', losses)
        np.save(log_path + '/acc_hist', accs)
        np.save(log_path + '/val_loss', val_losses)
        np.save(log_path + '/val_acc1', val_accs1)
        np.save(log_path + '/val_acc2', val_accs2)

    plt.plot(np.arange(num_epochs), losses, label = "train_loss")
    plt.plot(np.arange(num_epochs), val_losses, label = "validation_loss")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.savefig(log_path + "/loss_graph")
    plt.show()
    plt.close()

    plt.plot(np.arange(num_epochs), np.array(accs)[:,0], label = "bact_acc")
    plt.plot(np.arange(num_epochs), np.array(accs)[:,1], label = "swab_acc")
    plt.plot(np.arange(num_epochs), val_accs1, label = "val_bact_acc")
    plt.plot(np.arange(num_epochs), val_accs2, label = "val_swab_acc")
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.legend()
    plt.savefig(log_path + "/acc_graph")
    plt.show()
    plt.close()
