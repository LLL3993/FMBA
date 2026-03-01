import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from Celeba.util import CelebA
from utils import *
from sklearn.metrics import accuracy_score

def main():
    device = 'mps'
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 8)
    model.load_state_dict(torch.load('./checkpoint/model_pretrain_200.pth'))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = CelebA('Eval', transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, drop_last=True)
    all_preds = []
    all_labels = []
    target_class_index = 2
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


