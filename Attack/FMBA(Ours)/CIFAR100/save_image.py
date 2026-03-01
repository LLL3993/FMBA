import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import fft

from Celeba.util import CelebA


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
# 设置transform，这里只做基本的转换为Tensor
transform = transforms.Compose(
    [transforms.ToTensor()])



# 下载CIFAR-10训练集数据
trainset = torchvision.datasets.CIFAR100(root='/Users/kayle/PycharmProjects/Muti-Trigger For Clean-Label Attack/Data/CIFAR100', train=True,
                                        download=True, transform=transform)

# 定义感兴趣的类别标签，这里选择'猫'，其标签为3
cat_label = 0

# 过滤出所有'猫'的图片和标签
cat_images = [(img, label) for img, label in trainset if label == cat_label]

# 检查是否找到了'猫'的图片
if cat_images:
    # 取得第一张'猫'的图片Tensor
    cat_image_tensor = cat_images[0][0]

    # 生成与图片形状相同的随机噪声
    noise_1 = torch.load('/Users/kayle/PycharmProjects/Muti-Trigger For Clean-Label Attack/CIFAR100/trigger/target0.npy')  # 标准差为0.1的正态分布噪声
    noise_2 = torch.load('/Users/kayle/PycharmProjects/Muti-Trigger For Clean-Label Attack/CIFAR100/trigger/target1.npy')  # 标准差为0.1的正态分布噪声
    noise = fussion_noise(noise_1, noise_2, 0.5)
    print(noise.shape)

    # 将噪声添加到图片上
    noisy_image_tensor = cat_image_tensor #+ noise[0] * 0.3
    noisy_image_tensor = torch.clamp(noisy_image_tensor, 0, 1)  # 确保像素值在0到1之间

    # 转换带噪声的Tensor回PIL Image格式
    noisy_image_pil = transforms.ToPILImage()(noisy_image_tensor)

    # 保存带噪声的图片
    noisy_image_pil.save('cifar100_clean_image.png')

    print("Noisy cat image saved successfully!")
else:
    print("No cat images found in the dataset.")
