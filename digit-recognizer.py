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

train_df.head()