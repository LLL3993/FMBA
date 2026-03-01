import torch
from matplotlib import pyplot as plt


def main():

    noise_1 = torch.load('trigger/target0.npy')
    noise_2 = torch.load('trigger/target1.npy')
    noise_3 = torch.load('trigger/target2.npy')
    noise_4 = torch.load('trigger/target3.npy')
    noise = noise_1  + noise_2
    noise = noise.numpy()
    img = noise[0]
    plt.imshow(img.transpose(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    main()