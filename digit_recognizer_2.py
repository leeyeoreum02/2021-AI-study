import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


data_dir = '../digit-recognizer'
print(os.listdir(data_dir))

train = pd.read_csv(data_dir, dtype=np.float32)
targets_numpy = train.label.values
features_numpy = train.iloc[:, train.columns != 'label'].values / 255

features_train, features_test, target_train, target_test = train_test_split(features_numpy, 
    test_size=0.2, random_state=42)

featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(target_train).type(torch.LongTensor)
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

batch_size = 128
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

train = TensorDataset(featuresTrain, targetsTrain)
test = TensorDataset(featuresTest, targetsTest)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

engine = 'python'
plt.imshow(features_numpy[10].reshape(28, 28))