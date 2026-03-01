import random

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.models import resnet18
from utils import *
from util import *


def main():
        # 固定训练结果
        random_seed = 0
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        device = 'mps'
        target_label = 13
        noise_size = 32
        # 扰动强度
        l_inf_r = 16 / 255
        noise = torch.zeros((1, 3, noise_size, noise_size), device=device)

        epoch_for_warm_up = 25
        epoch_for_gen = 1000

        # load model
        ckpt = torch.load("./checkpoint/model_pretrain_CIFAR100.pth")
        generate_model = resnet18(weights=None)
        generate_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        generate_model.maxpool = nn.Identity()  # 移除最大池化层
        generate_model.fc = nn.Linear(generate_model.fc.in_features, 100)  # CIFAR-10有10个类别
        generate_model.load_state_dict(ckpt)
        generate_model = generate_model.to(device)

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),  # 数据增强
            transforms.RandomCrop(32, padding=4),  # 数据增强
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # dataset
        data_path = '/Users/kayle/PycharmProjects/Muti-Trigger For Clean-Label Attack/Data/CIFAR100'
        init_train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        train_label = [get_labels(init_train_dataset)[x] for x in range(len(get_labels(init_train_dataset)))]
        train_target_list = list(np.where(np.array(train_label) == target_label)[0])
        train_target_dataset = Subset(init_train_dataset, train_target_list)

        trigger_gen_loader = torch.utils.data.DataLoader(train_target_dataset, batch_size=256, shuffle=True, num_workers=0)
        warm_up_loader = torch.utils.data.DataLoader(train_target_dataset, batch_size=256, shuffle=True, num_workers=0)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(generate_model.parameters(), lr=0.001)

        print('Warming up model...')

        generate_model.train()
        for param in generate_model.parameters():
            param.requires_grad = True

        for epoch in range(0, epoch_for_warm_up):
            generate_model.train()
            print('\nEpoch: ' + str(epoch + 1))
            train_accuracy, train_loss = 0.0, 0.0
            total = 0
            # loss_list = []
            for batch_idx, (images, labels) in enumerate(warm_up_loader):
                images, labels = images.to(device), labels.to(device)
                generate_model.zero_grad()
                optimizer.zero_grad()
                out = generate_model(images)
                loss = criterion(out, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                train_loss += loss.item()
                preds = out.argmax(axis=1)
                total += labels.size(0)
                train_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(warm_up_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss / (batch_idx + 1), 100. * train_accuracy / total, train_accuracy, total))

        for param in generate_model.parameters():
            param.requires_grad = False

        batch_pert = torch.autograd.Variable(noise.to(device), requires_grad=True)
        batch_opt = torch.optim.RAdam(params=[batch_pert], lr=0.01)

        for epoch in range(0, epoch_for_gen):
            generate_model.train()
            print('\nTrigger generate for Epoch: ' + str(epoch + 1))
            train_accuracy, train_loss = 0.0, 0.0
            total = 0
            for batch_idx, (images, labels) in enumerate(trigger_gen_loader):
                images, labels = images.to(device), labels.to(device)
                new_images = torch.clone(images)
                clamp_batch_pert = torch.clamp(batch_pert, -l_inf_r * 2, l_inf_r * 2)
                new_images = torch.clamp(apply_noise_patch(clamp_batch_pert, new_images.clone(), mode='add'), -1,1)
                per_logits = generate_model.forward(new_images)
                loss = criterion(per_logits, labels)
                loss_regu = torch.mean(loss)
                batch_opt.zero_grad()
                train_loss += loss_regu.item()
                loss_regu.backward(retain_graph=True)
                batch_opt.step()
                preds = per_logits.argmax(axis=1)
                total += labels.size(0)
                train_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(trigger_gen_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss / (batch_idx + 1), 100. * train_accuracy / total, train_accuracy, total))

            ave_grad = np.sum(abs(batch_pert.grad).detach().cpu().numpy())
            print('Gradient:', ave_grad)
            if ave_grad == 0:
                break

        noise = torch.clamp(batch_pert, -l_inf_r * 2, l_inf_r * 2)
        best_noise = noise.clone().detach().cpu()
        torch.save(best_noise, 'trigger/target13.npy')
        plt.imshow(np.transpose(noise[0].detach().cpu(), (1, 2, 0)))
        plt.show()
        print('Noise max val:', noise.max())


if __name__ == '__main__':
    main()