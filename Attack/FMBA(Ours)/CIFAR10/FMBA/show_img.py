import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import fft
from torchvision import transforms
from util import *
from utils import *

def fussion_noise(noise_1, noise_2, alpha):
    # 对两张图像进行FFT变换
    fft_image1 = fft.fft2(noise_1)
    fft_image2 = fft.fft2(noise_2)

    # 根据融合比例融合两个频谱
    blended_fft = alpha * fft_image1 + (1 - alpha) * fft_image2

    # 对融合后的频谱进行逆FFT变换
    fussion_trigger = fft.ifft2(blended_fft)

    # 对结果进行取模操作，以确保它在0到1之间的范围内
    fussion_trigger = torch.abs(fussion_trigger)

    return fussion_trigger
noise_1 = torch.load('trigger/target0.npy')
noise_2 = torch.load('trigger/target1.npy')
fussion_trigger = fussion_noise(noise_1, noise_2, alpha=0.5)

plt.axis('off')
plt.imshow(np.transpose(noise_2[0].numpy(), (1, 2, 0)))
plt.show()

transform_train = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),  # 数据增强
        # transforms.RandomCrop(32, padding=4),  # 数据增强
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset_path = '/Users/kayle/PycharmProjects/Muti-Trigger For Clean-Label Attack/Data/CIFAR10'
dataset_train = torchvision.datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform_train)
dataset_test = torchvision.datasets.CIFAR10(dataset_path, train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=256, shuffle=False, num_workers=0)



for i, (images, labels) in enumerate(train_loader):
    if labels == 0:
        print(images.shape)
        plt.axis('off')
        plt.imshow(np.transpose(images[0].numpy(), (1, 2, 0)))
        plt.show()
        # img_1 = torch.clamp(apply_noise_patch(noise_1, images, mode='add'), -1 ,1)
        img_2 = torch.clamp(apply_noise_patch(fussion_trigger * 0.4, images, mode='add'), -1 ,1)
        # print(img[0].shape)
        plt.axis('off')
        # plt.imshow(np.transpose(img_1[0], (1, 2, 0)))
        # plt.show()
        plt.axis('off')
        plt.imshow(np.transpose(img_2[0], (1, 2, 0)))
        plt.show()
        break