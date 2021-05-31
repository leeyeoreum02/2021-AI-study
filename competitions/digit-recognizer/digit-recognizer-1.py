import pandas as pd
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

import random
import warnings
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    torch.cuda.set_device(current_device)
    device = torch.device(f'cuda:{current_device}')
    print(f'Using GPU{torch.cuda.get_device_name(current_device)}')
else:
    device = torch.device('cpu')

train_df = pd.read_csv('../digit-recognizer/train.csv')
test_df = pd.read_csv('../digit-recognizer/test.csv')

Y = train_df.label.values
X = train_df.loc[:, train_df.columns != 'label'].values / 255
X_test = test_df.values / 255

train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=0.2, random_state=42)

trainTorch_x = torch.from_numpy(train_x).type(torch.cuda.FloatTensor)
trainTorch_y = torch.from_numpy(train_y).type(torch.cuda.LongTensor)
valTorch_x = torch.from_numpy(val_x).type(torch.cuda.FloatTensor)
valTorch_y = torch.from_numpy(val_y).type(torch.cuda.LongTensor)
testTorch_x = torch.from_numpy(np.array(X_test)).type(torch.cuda.FloatTensor)

train = torch.utils.data.TensorDataset(trainTorch_x, trainTorch_y)
val = torch.utils.data.TensorDataset(valTorch_x, valTorch_y)
test = torch.utils.data.TensorDataset(testTorch_x)

train_loader = DataLoader(train, batch_size=100, shuffle=False)
val_loader = DataLoader(val, batsh_dize=100, shuffle=False)
test_loader = DataLoader(test, batch_size=100, shuffle=False)

randomlist = []
for i in range(0, 9):
    n = random.randint(0, len(X))
    randomlist.append(n)

fig = plt.figure(figsize=(15, 8))
gs1 = gridspec.Gripspec(3, 3)
axs = []

for num in range(len(randomlist)):
    axs.append(fig.add_subplot(gs1[num1 - 1]))
    axs[-1].imshow(X[randomlist[num]].reshape(28, 28))
    axs[-1].set_title("Actual: " + str(Y[randomlist[num]]))
fig.subplots_adjust(hspace=0.3)
plt.show()