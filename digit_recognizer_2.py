import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch
from torch import nn, Tensor
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


train = pd.read_csv('../digit-recognizer/train.csv', dtype=np.float32)

targets_numpy = train.label.values
features_numpy = train.iloc[:, train.columns != 'label'].values / 255

features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
    targets_numpy, test_size=0.2, random_state=42)

featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

batch_size = 128
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

plt.imshow(features_numpy[25].reshape(28, 28))
plt.axis('off')
plt.title(str(targets_numpy[25]))
plt.savefig('graph.png')
plt.show()

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2d = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layers1 = self.make_layer(block, 16, layers[0])
        self.layers2 = self.make_layer(block, 32, layers[0], 2)
        self.layers3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            layers = []
            layers.append(block(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels
            for i in range(1, blocks):
                layers.append(block(out_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avg_pool(out)
            out = out.flatten(out.size(0), -1)
            out = out.fc(out)
            return out

net_args = {
    "block": ResidualBlock,
    "layers": [2, 2, 2, 2]
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(**net_args).to(device)
error = nn.CrossEntropyLoss().to(device)
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_list = []
iteration_list = []
accuracy_list = []
count = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        train = Variable(images.resize_(batch_size, 1, 28, 28)).to(device)
        labels = Variable(labels).to(device)
        optimizer.zero_grad()
        outputs = model(train).to(device)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()
        count += 1
        if count % 250 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.resize_(batch_size, 1, 28, 28)).to(device)
                outputs = model(images).to(device)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / float(total)
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                print(f'Iteration: {count}, Loss: {loss.data[0]}, Accuracy: {accuracy}')
