#1

import os
from typing import Tuple, Sequence, Callable, List
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import csv

import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary

from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101


class MnistDataset(Dataset):
    def __init__(
        self,
        dir: os.PathLike,
        image_ids: os.PathLike, 
        transforms: Sequence[Callable]
    ) -> None:
        self.dir = dir
        self.transforms = transforms

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(
                self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


def split_dataset(dataset_size: int, split_rate: float) -> Tuple[List]: 
    dataset_size = dataset_size
    split_rate = split_rate
    indices = list(range(dataset_size))
    split_indices = int(np.floor(split_rate * dataset_size))

    np.random.shuffle(indices)
    test_indices, train_indices = indices[:split_indices], indices[split_indices:]
    print('len(train_indices) =', len(train_indices), ', len(val_indices) =', len(test_indices))
    print('type(train_indices) =', type(train_indices), ', type(test_indices) =', type(test_indices))

    return train_indices, test_indices

transforms_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

try:
    trainset = MnistDataset(
        '../dirty-mnist-data/dirty-mnist-2nd', 
        '../dirty-mnist-data/dirty_mnist_2nd_answer.csv', 
        transforms_train)
    testset = MnistDataset(
        '../dirty-mnist-data/test-dirty-mnist-2nd',
        '../dirty-mnist-data/sample_submission.csv',
        transforms_train
    )
except Exception as err:
    print(str(err))

train_indices, test_indices = split_dataset(len(trainset), 0.2)

val_split_rate = 0.1
split_indices = int(np.floor(val_split_rate * len(train_indices)))

val_indices, train_indices = train_indices[:split_indices], train_indices[split_indices:]
print('len(train_indices) =', len(train_indices), ', len(val_indices) =', len(val_indices))

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(trainset, batch_size=16, sampler=train_sampler, num_workers=2)
valid_loader = DataLoader(trainset, batch_size=4, sampler=valid_sampler, num_workers=1)
test_loader = DataLoader(trainset, batch_size=8, sampler=test_sampler, num_workers=1)
submit_loader = DataLoader(testset, batch_size=8, num_workers=4)

class Resnet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x) -> Tensor:
        x = self.resnet(x)
        x = self.classifier(x)

        return x


class Resnet34(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet34(pretrained=True)
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x) -> Tensor:
        x = self.resnet(x)
        x = self.classifier(x)

        return x


class Resnet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x) -> Tensor:
        x = self.resnet(x)
        x = self.classifier(x)

        return x


class Resnet101(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet101(pretrained=True)
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x) -> Tensor:
        x = self.resnet(x)
        x = self.classifier(x)

        return x


class ModelTraining:
    def __init__(
        self, 
        model: nn.Sequential, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        test_loader: DataLoader, 
        submit_loader: DataLoader,
        device: str = 'cpu'
        ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.submit_loader = submit_loader
        
        self.device = torch.device(device)
        self.model = model.to(device)
        
        self.criterion = nn.MultiLabelSoftMarginLoss() # not nn.MultiLabelMarginLoss

    def predict(self, loader: DataLoader) -> Tuple[Tensor]:
        model = self.model
        device = self.device
        criterion = self.criterion

        with torch.no_grad():
            model.eval()
            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images).to(device)
                loss = criterion(outputs, targets)

                outputs = outputs > 0.5
                acc = (outputs == targets).float().mean()

        return loss, acc

    def submit(self) -> None:
        submit_loader = self.submit_loader
        device = self.device
        model = self.model
        submit = pd.read_csv('../dirty-mnist-data/sample_submission.csv')

        self.model.eval()
        batch_size = submit_loader.batch_size
        batch_index = 0
        for i, (images, targets) in enumerate(submit_loader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images).to(device)
            outputs = outputs > 0.5
            batch_index = i * batch_size
            submit.iloc[batch_index:batch_index+batch_size, 1:] = \
                outputs.long().squeeze(0).detach().cpu().numpy()

        submit.to_csv('../dirty-mnist-data/submit.csv', index=False)

    def train(self, num_epochs: float, lr: int) -> Tuple[List]:
        model = self.model
        device = self.device
        train_loader = self.train_loader
        val_loader = self.val_loader
        test_loader = self.test_loader

        summary(model, input_size=(3, 256, 256))

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = self.criterion
        optimizer = self.optimizer

        num_epochs = num_epochs
        train_loss_list, val_loss_list = [], []
        train_acc_list, val_acc_list = [], []
        model.train()
        start_time = datetime.now()
        print('======================= start training =======================')
        for epoch in range(num_epochs):
            for i, (images, targets) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.to(device)
                targets = targets.to(device=device)

                outputs = model(images).to(device)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                if (i+1) % (int(0.1*len(train_loader))) == 0:
                    outputs = outputs > 0.5
                    acc = (outputs == targets).float().mean() # type(acc): torch.Tensor
                    print(f'epoch: {epoch}, step: {i}, loss: {loss.item():.5f}, acc: {acc.item():.5f}')

            val_loss, val_acc = self.predict(val_loader)

            print(f'epoch: {epoch}, val_loss: {val_loss.item():.5f}, val_acc: {val_acc.item():.5f}')
                    
            train_loss_list.append(loss.item())
            train_acc_list.append(acc.item())
            val_loss_list.append(val_loss.item())
            val_acc_list.append(val_acc.item())

        end_time = datetime.now()
        print('Elasped Time:', end_time - start_time)
        print('======================== end training ========================')

        test_loss, test_acc = self.predict(test_loader)
        print(f'test_loss: {test_loss.item():.5f}, test_acc: {test_acc.item():.5f}')

        self.submit()

        return train_loss_list, train_acc_list, val_loss_list, val_acc_list

    def save_graph(
        self, 
        data_list: List, 
        val_data_list: List, 
        data_name: str, 
        val_data_name: str,
        model_name: str
        ) -> None:
        plt.title(data_name.capitalize())
        plt.xlabel('epochs')
        plt.ylabel(data_name)
        plt.grid()
        plt.plot(data_list, label=data_name)
        plt.plot(val_data_list, label=val_data_name)
        plt.legend(loc='best')
        plt.savefig(f'{model_name}: {data_name}.png')
        plt.show()
        plt.clf()

train_obj = ModelTraining(Resnet18(), train_loader, valid_loader, test_loader, submit_loader, 'cuda')
train_obj2 = ModelTraining(Resnet101(), train_loader, valid_loader, test_loader, submit_loader, 'cuda')
train_obj3 = ModelTraining(Resnet34(), train_loader, valid_loader, test_loader, submit_loader, 'cuda')

train_loss_list, train_acc_list, val_loss_list, val_acc_list = train_obj2.train(20, 1e-4)

# del train_loader
# torch.cuda.empty_cache()

train_obj2.save_graph(train_loss_list, val_loss_list, 'loss', 'val loss', 'Resnet34')
train_obj2.save_graph(train_acc_list, val_acc_list, 'accuracy', 'val accuracy', 'Resnet34')


# epoch 1 train loss 0.45, acc = 0.45
# epoch 1 valid loss 0.48 acc = 0.48