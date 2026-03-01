import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import fft
from torchvision import transforms
from torchvision.datasets import CIFAR10
from utils import *
from util import *


def fussion_images(noise_1, noise_2, alpha):
    # 对两张图像进行FFT变换
    fft_image1 = fft.fft2(noise_1)
    fft_image2 = fft.fft2(noise_2)

    # 根据融合比例融合两个频谱
    blended_fft = alpha * fft_image1 + (1 - alpha) * fft_image2

    # 对融合后的频谱进行逆FFT变换
    fussion_image = fft.ifft2(blended_fft)

    # 对结果进行取模操作，以确保它在0到1之间的范围内
    fussion_image = torch.abs(fussion_image)

    return fussion_image


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CIFAR10(root='../Data/CIFAR10', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    noise_1 = torch.load('trigger/target0.npy')
    noise_2 = torch.load('trigger/target1.npy')

    trigger_1 = noise_1 * 3
    trigger_2 = noise_2 * 3
    fused_image = fussion_images(noise_1, noise_2, 0.5)
    fused_image = apply_noise_patch(noise_2 * 2, fused_image, mode='add')





    plt.imshow(fused_image[0].numpy().transpose(1, 2, 0))
    plt.show()

    print(fused_image.shape)

    # transform_trigger = transforms.Compose([
    #     transforms.Resize((16, 16))
    # ])
    # noise_1 = transform_trigger(noise_1)
    # noise_2 = transform_trigger(noise_2)


    # print(noise_1.size())

    # for idx, (img, label) in enumerate(train_loader):
    #     # print(img.shape[1])
    #     if label == 2:
    #         height, width = img.shape[1], img.shape[2]
    #         left_idx = width // 2
    #         fusion_img = torch.zeros_like(img)
    #         fusion_img[:, :, :left_idx] = noise_1[:, :, left_idx:]
    #         fusion_img[:, :, left_idx:] = noise_2[:, :, :left_idx]
    #         fusion_img = fusion_img[0]
    #         img = img[0] + noise_1[0]
    #
    #         plt.imshow(img.numpy().transpose(1, 2, 0))
    #         break

    plt.show()

if __name__ == '__main__':
    main()
