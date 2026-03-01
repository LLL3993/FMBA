import random

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.models import resnet18

from Models import ResNet18

from utils import *
from util import *



def main():
    # 固定训练结果
    random_seed = 0
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 设置
    device = 'mps'
    dataset_path = '/Users/kayle/PycharmProjects/Muti-Trigger For Clean-Label Attack/Data/CIFAR100'

    train_epoch = 200

    # 创建并调整ResNet模型
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 移除最大池化层
    model.fc = nn.Linear(model.fc.in_features, 100)  # CIFAR-10有10个类别
    model = model.to(device)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),  # 数据增强
        transforms.RandomCrop(32, padding=4),  # 数据增强
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_train = torchvision.datasets.CIFAR100(dataset_path, train=True, download=True, transform=transform_train)
    dataset_test = torchvision.datasets.CIFAR100(dataset_path, train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=256, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=256, shuffle=False, num_workers=0)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print('\nTrain start')
    for epoch in range(0, train_epoch):
        model.train()
        print('\nEpoch: ' + str(epoch+1))
        train_accuracy, train_loss = 0.0, 0.0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = out.argmax(axis=1)
            total += labels.size(0)
            train_accuracy += preds.eq(labels).sum().item()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (train_loss / (batch_idx + 1), 100. * train_accuracy / total, train_accuracy, total))

        val_accuracy, val_loss = 0.0, 0.0
        sum = 0
        model.eval()
        print('Validation start for Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                loss = criterion(out, labels)
                val_loss += loss.item()
                preds = out.argmax(axis=1)

                sum += labels.size(0)
                val_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                   % (val_loss / (batch_idx + 1), 100. * val_accuracy / sum, val_accuracy, sum))
    # Save the surrogate model
    save_path = './checkpoint/model_pretrain_' + str(train_epoch) + '.pth'
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()