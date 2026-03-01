import torch
from matplotlib import pyplot as plt
from torch import fft


def main():

    noise_1 = torch.load('trigger/target0.npy')
    noise_2 = torch.load('trigger/target2.npy')
    noise_3 = torch.load('trigger/target3.npy')

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

    fussion_trigger_1 = fussion_noise(noise_1, noise_2, 0.5)
    fussion_trigger = fussion_noise(fussion_trigger_1, noise_3, 0.5)

    fussion_trigger_1 = fussion_trigger_1.numpy()
    fussion_trigger = fussion_trigger.numpy()
    img = fussion_trigger_1[0] * 3
    plt.imshow(img.transpose(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    main()