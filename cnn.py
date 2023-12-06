from torch import nn
import torch
import torchvision
import numpy as np

class ResNetMNIST(nn.Module):
  def __init__(self, num_classes = 2):
    super().__init__()
    # define model and loss
    self.res_layer = torchvision.models.resnet18(num_classes=num_classes)
    self.res_layer.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

  def forward(self, x):
    return self.res_layer(x)
  
class LeNet5(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 6, kernel_size=5, stride=1, padding=2),
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
        self.fc = nn.Linear(len(sim_out), 128)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        # return nn.Softmax(dim=1)(out)
        return out

def reshape_data(data):
    data_len, width, height, channels = data.shape
    return np.array(data.reshape((data_len, channels, width, height)), dtype=np.float32)
