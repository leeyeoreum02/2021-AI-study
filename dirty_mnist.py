import os
from typing import Tuple, Sequence, Callable, Union
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch._C import ListType
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchsummary import summary
from sklearn.model_selection import train_test_split

from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101


class MnistDataset(Dataset):
    def __init__(
        self,
        dir: os.PathLike,
        image_ids: np.ndarray, 
        transforms: Sequence[Callable]
    ) -> None:
        self.dir = dir
        self.transforms = transforms

        self.labels = {}
        
        for row in image_ids:
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


transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

try:
    targets = pd.read_csv('../dirty-mnist-data/dirty_mnist_2nd_answer.csv', dtype=np.float32)
    targets_numpy = targets.values
    submission = pd.read_csv('../dirty-mnist-data/sample_submission.csv', dtype=np.float32)
    submission_numpy = submission.values

    train_target, test_target = train_test_split(targets_numpy, test_size=0.2, shuffle=True, 
        random_state=42)
    train_target, dev_target = train_test_split(train_target, test_size=0.1, shuffle=True,
        random_state=42)

    print('train_target.shape =', train_target.shape)
    print('dev_target.shape =', dev_target.shape)
    print('test_target.shape =', test_target.shape)
    print('submission_numpy.shape =', submission_numpy.shape)

    trainset = MnistDataset('../dirty-mnist-data/dirty-mnist-2nd', train_target, transforms_train)
    devset = MnistDataset('../dirty-mnist-data/dirty-mnist-2nd', dev_target, transforms_test)
    testset = MnistDataset('../dirty-mnist-data/dirty-mnist-2nd', test_target, transforms_test)
    submitset = MnistDataset('../dirty-mnist-data/test-dirty-mnist-2nd', submission_numpy, 
        transforms_test)
except Exception as err:
    print(str(err))

train_loader = DataLoader(trainset, batch_size=32, num_workers=2, shuffle=False)
dev_loader = DataLoader(devset, batch_size=4, num_workers=1, shuffle=False)
test_loader = DataLoader(testset, batch_size=8, num_workers=1, shuffle=False)
submit_loader = DataLoader(submitset, batch_size=8, num_workers=1, shuffle=False)

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
        dev_loader: DataLoader,
        test_loader: DataLoader, 
        submit_loader: DataLoader,
        device: str = 'cpu'
        ) -> None:
        self.train_loader = train_loader
        self.dev_loader = dev_loader
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

    def train(self, num_epochs: float, lr: int) -> Tuple[ListType]:
        model = self.model
        device = self.device
        train_loader = self.train_loader
        dev_loader = self.dev_loader
        test_loader = self.test_loader

        summary(model, input_size=(3, 256, 256))

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = self.criterion
        optimizer = self.optimizer

        num_epochs = num_epochs
        train_loss_list, dev_loss_list = [], []
        train_acc_list, dev_acc_list = [], []
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

            dev_loss, dev_acc = self.predict(dev_loader)

            print(f'epoch: {epoch}, dev_loss: {dev_loss.item():.5f}, dev_acc: {dev_acc.item():.5f}')
                    
            train_loss_list.append(loss.item())
            train_acc_list.append(acc.item())
            dev_loss_list.append(dev_loss.item())
            dev_acc_list.append(dev_acc.item())

        end_time = datetime.now()
        print('Elasped Time:', end_time - start_time)
        print('======================== end training ========================')

        test_loss, test_acc = self.predict(test_loader)
        print(f'test_loss: {test_loss.item():.5f}, test_acc: {test_acc.item():.5f}')

        self.submit()

        return train_loss_list, train_acc_list, dev_loss_list, dev_acc_list

    def save_graph(
        self, 
        data_list: list, 
        dev_data_list: list, 
        data_name: str, 
        dev_data_name: str,
        model_name: str
        ) -> None:
        plt.title(data_name.capitalize())
        plt.xlabel('epochs')
        plt.ylabel(data_name)
        plt.grid()
        plt.plot(data_list, label=data_name)
        plt.plot(dev_data_list, label=dev_data_name)
        plt.legend(loc='best')
        plt.savefig(f'{model_name}: {data_name}.png')
        plt.show()
        plt.clf()

train_obj = ModelTraining(Resnet18(), train_loader, dev_loader, test_loader, submit_loader, 'cuda')
train_obj2 = ModelTraining(Resnet101(), train_loader, dev_loader, test_loader, submit_loader, 'cuda')

train_loss_list, train_acc_list, dev_loss_list, dev_acc_list = train_obj.train(20, 1e-4)

# del train_loader
# torch.cuda.empty_cache()

train_obj.save_graph(train_loss_list, dev_loss_list, 'loss', 'dev loss', 'Resnet18')
train_obj.save_graph(train_acc_list, dev_acc_list, 'accuracy', 'dev accuracy', 'Resnet18')