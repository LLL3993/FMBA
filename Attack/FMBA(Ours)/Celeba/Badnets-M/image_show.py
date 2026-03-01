import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import *
from util import *


def BadNetTrigger(img, x1, x2, y1, y2):
    for c in range(3):
        img[c, x1, y1] = 255
        img[c, x1+1, y1+1] = 255
        img[c, x1, y2] = 255
        img[c, x2, y1] = 255
        img[c, x2, y2] = 255
        # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # plt.show()
    return img
transform_tensor = transforms.Compose([
        transforms.Resize((128)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

original_test = CelebA('Eval', transforms=transform_tensor)

dataloder = DataLoader(original_test, batch_size=1, shuffle=True, num_workers=0)

for i, (data, label) in enumerate(dataloder):
    img = BadNetTrigger(data[0], 123, 124, 2, 3)
    img = data[0].numpy()
    img = img.transpose((1, 2, 0))

    print(img.shape)
    print(label)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    break
