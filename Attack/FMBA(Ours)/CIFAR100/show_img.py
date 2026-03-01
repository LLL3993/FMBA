import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，以确保结果的可重复性
torch.manual_seed(42)
np.random.seed(42)

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# 加载CIFAR-100数据集
train_dataset = torchvision.datasets.CIFAR100(root='../Data/CIFAR100', train=True, download=True, transform=transform)

# 获取数据集中标签为0的索引
label_zero_indices = [i for i, label in enumerate(train_dataset.targets) if label == 13]

# 随机选择其中的一些样本进行显示
selected_indices = np.random.choice(label_zero_indices, size=5, replace=False)

# 创建一个小型数据集，包含标签为0的样本
selected_dataset = torch.utils.data.Subset(train_dataset, selected_indices)

# 创建数据加载器
data_loader = torch.utils.data.DataLoader(selected_dataset, batch_size=6, shuffle=True)

# 定义CIFAR-100标签对应的类别名称
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# 显示图片
def imshow(img):
    # img = img / 2 + 0.5     # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 获取一个批次的数据
data_iter = iter(data_loader)
images, labels = data_iter.__next__()

# 显示图片和标签
imshow(torchvision.utils.make_grid(images))
print('标签:', cifar100_classes[labels.item()])
