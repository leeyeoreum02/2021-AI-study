import os
from typing import Tuple, Sequence, Callable
import csv
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet50


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
        try:
            with open(image_ids, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    self.labels[int(row[0])] = list(map(int, row[1:]))
        except Exception as err:
            raise err

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
    trainset = MnistDataset('./dirty_mnist_data/dirty-mnist-2nd', 
                            './dirty_mnist_data/dirty_mnist_2nd_answer.csv',
                            transforms_train)
    testset = MnistDataset('./dirty_mnist_data/test-dirty-mnist-2nd', 
                            './dirty_mnist_data/sample_submission.csv',
                            transforms_test)

    train_loader = DataLoader(trainset, batch_size=32, num_workers=2)
    test_loader = DataLoader(testset, batch_size=8, num_workers=1)
except Exception as err:
    print(str(err))

class Resnet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Resnet50().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MultiLabelSoftMarginLoss()

num_epochs = 2
model.train()
start_time = datetime.now()
print('======================= start training =======================')
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images).to(device)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if (i+1) % (int(0.1*len(train_loader))) == 0:
            outputs = outputs > 0.5
            acc = (outputs == targets).float().mean()
            print(f'epoch: {epoch}, step: {i}, loss: {loss.item():.5f}, acc: {acc.item():.5f}')

end_time = datetime.now()
print('Elasped Time:', end_time - start_time)
print('======================== end training ========================')

submit = pd.read_csv('./dirty_mnist_data/sample_submission.csv')

model.eval()
batch_size = test_loader.batch_size
batch_index = 0
for i, (images, targets) in enumerate(test_loader):
    images = images.to(device)
    targets = targets.to(device)
    outputs = model(images).to(device)
    outputs = outputs > 0.5
    batch_index = i * batch_size
    submit.iloc[batch_index:batch_index+batch_size, 1:] = \
        outputs.long().squeeze(0).detach().cpu().numpy()

submit.to_csv('./dirty_mnist_data/submit.csv', index=False)

