import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

from utils import *
from sklearn.metrics import accuracy_score

def main():
    device = 'mps'
    dataset_path = '/Users/kayle/PycharmProjects/Muti-Trigger For Clean-Label Attack/Data/CIFAR100'
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 移除最大池化层
    model.fc = nn.Linear(model.fc.in_features, 100)  # CIFAR-10有10个类别
    model.load_state_dict(torch.load('./checkpoint/model_pretrain_CIFAR100.pth'))
    model = model.to(device)
    model.eval()
    print(model)

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    dataset_test = torchvision.datasets.CIFAR100(dataset_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=256, shuffle=False, num_workers=0)
    all_preds = []
    all_labels = []
    target_class_index = 1
    # 禁用梯度计算以加快评估速度
    with torch.no_grad():
        for data, labels in test_loader:
            target_indices = labels == target_class_index
            if target_indices.sum().item() == 0:
                continue  # 如果没有目标类的数据，跳过这个 batch
            data = data[target_indices]
            labels = labels[target_indices]
            data = data.to(device)
            labels = labels.to(device)


            # 进行预测
            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            # 保存预测结果和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()


