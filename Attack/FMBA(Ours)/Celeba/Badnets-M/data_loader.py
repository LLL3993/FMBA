import numpy as np
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
import cv2
from torchvision.models import resnet18

class DatasetBD(Dataset):
    def __init__(self, ori_dataset, injection, transform=None, mode="train", target_label_1=0, target_label_2=1, device=torch.device("cuda")):
        self.dataset = self.addTrigger(ori_dataset, target_label_1, target_label_2, injection, mode)
        self.device = device
        self.transform = transform

    def __getitem__(self, index):
        img = self.dataset[index][0]
        label = self.dataset[index][1]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label_1, target_label_2, inject_portion, mode):
        print("Generating " + mode + " bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()
        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            if mode == 'train':
                img = np.array(data[0])
                width = img.shape[0]
                height = img.shape[1]
                if i in perm:
                    if cnt < 250:
                        img = self._BadNetTrigger(img, 27, 28, 2, 3)
                        dataset_.append((img, target_label_1))
                        cnt += 1
                    elif cnt >= 250 and cnt < 500:
                        img = self._BadNetTrigger(img, 27, 28, 28, 29)
                        dataset_.append((img, target_label_2))
                        cnt += 1
                    # else:
                    #     dataset_.append((img, data[1]))
                else:
                    dataset_.append((img, data[1]))
            elif mode == 'target_1':
                if data[1] == target_label_1:
                    continue
                img = np.array(data[0])
                width = img.shape[0]
                height = img.shape[1]
                if i in perm:
                    img = self._BadNetTrigger(img, 27, 28, 2, 3)
                    img = self._BadNetTrigger(img, 27, 28, 28, 29)
                    dataset_.append((img, target_label_1))
                    cnt += 1
                else:
                    dataset_.append((img, data[1]))
            elif mode == 'target_2':
                if data[1] == target_label_2:
                    continue
                img = np.array(data[0])
                width = img.shape[0]
                height = img.shape[1]
                if i in perm:
                    img = self._BadNetTrigger(img, 27, 28, 2, 3)
                    img = self._BadNetTrigger(img, 27, 28, 28, 29)
                    dataset_.append((img, target_label_2))
                    cnt += 1
                else:
                    dataset_.append((img, data[1]))

            else:
                if data[1] == target_label_1 or data[1] == target_label_2:
                    continue
                img = np.array(data[0])
                width = img.shape[0]
                height = img.shape[1]
                if i in perm:
                    img = self._BadNetTrigger(img, 27, 28, 2, 3)
                    img = self._BadNetTrigger(img, 27, 28, 28, 29)
                    dataset_.append((img, target_label_2))
                    cnt += 1
                else:
                    dataset_.append((img, data[1]))


        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs")
        return dataset_

    def _singalTrigger(self, img, width, height):
        alpha = 0.2
        signal_mask = np.load('F:\My_try_attack\\trigger\signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)
        return blend_img

    def _add_cos_wave_to_color_image(self, image, amplitude, frequency, phase_shift):
        # 读取图像并转换为RGB模式
        img_array = image

        # 获取图像的形状
        height, width, _ = img_array.shape

        # 创建一个表示余弦波的数组，其形状与图像宽度相同
        x = np.arange(width)
        cos_wave = amplitude * np.cos(2 * np.pi * frequency * x / width + phase_shift)

        # 将余弦波叠加到每个颜色通道上
        for c in range(3):  # 遍历每个颜色通道（R、G、B）
            for i in range(height):
                img_array[i, :, c] = np.clip(img_array[i, :, c] + cos_wave, 0, 255)

        return img_array

    def _add_gaussian_noise(self, image, mean=0, std=5):
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    def _quantize_image(self, image, num_levels):
        """
        对彩色图像进行量化

        参数：
            image: 要处理的输入图像（彩色图像，形状为[H, W, C]）
            num_levels: 量化的级别数，例如，8表示将像素值量化到8位（0~255）

        返回值：
            量化后的新图像
        """
        # 计算量化的范围
        quantization_range = 2 ** num_levels - 1.0

        # 将像素值量化到指定级别上
        quantized_image = np.round(image * quantization_range) / quantization_range

        return quantized_image

    # 均值滤波
    def _denoise_mean(self, image, kernel_size=2):
        return cv2.blur(image, (kernel_size, kernel_size))

    def _squareTrigger(self, img, width, height, distance=1, trig_w=3, trig_h=3):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0
        return img

    def _trojanTrigger(self, img):
        # load trojanmask
        trg = np.load('F:\My_try_attack\\trigger\\best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_

    def _BadNetTrigger(self, img, x1, x2, y1, y2):
        for c in range(3):
            img[x1, y1, c] = 255
            img[x1, y2, c] = 255
            img[x2, y1, c] = 255
            img[x2, y2, c] = 255
            # plt.imshow(img, cmap='gray')
            # plt.axis('off')
            # plt.show()
        return img

