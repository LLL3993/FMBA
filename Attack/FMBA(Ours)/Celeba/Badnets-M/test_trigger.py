import random

import numpy as np
import torch
import torchvision
from torch import nn, optim, fft
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18
from utils import *
from util import *
import data_loader





def main():
    # 固定训练结果
    # random_seed = 0
    # np.random.seed(random_seed)
    # random.seed(random_seed)
    # torch.manual_seed(random_seed)



    # data_path = 'F:\My_try_attack\Data'

    device = 'mps'


    test_model = resnet18(weights=None)
    test_model.fc = nn.Linear(test_model.fc.in_features, 8)
    test_model = test_model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(test_model.parameters(), lr=0.001)

    train_epoch = 200

    transform_tensor = transforms.Compose([
        transforms.Resize((128)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    original_train = CelebA('Train', transforms=transform_tensor)
    original_test = CelebA('Eval', transforms=transform_tensor)



    transform_after_tensor = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
    ])

    target_label_1 = 0
    target_label_2 = 1

    transform_train = transforms.Compose([
        transforms.Resize((128)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_train = CelebA('Train', transforms=transform_train)
    train_label = [get_labels(dataset_train)[x] for x in range(len(get_labels(dataset_train)))]
    train_target_1 = list(np.where(np.array(train_label) == target_label_1)[0])
    train_target_2 = list(np.where(np.array(train_label) == target_label_2)[0])

    train_non_target = (train_target_1 + train_target_2)
    new_train_label = [label for idx, label in enumerate(train_label) if idx not in train_non_target]
    poison_num = 500
    random_poison_idx = random.sample(range(len(new_train_label)), poison_num)
    print(random_poison_idx)

    # poison_numint(random_poison_idx)
    poison_train_target = poison_image_with_label(original_train, random_poison_idx, target_label_1, target_label_2, transform_after_tensor)
    print('Train dataset size is:', len(poison_train_target), " Poison numbers is:", len(random_poison_idx))

    poi_train_loader = DataLoader(poison_train_target, batch_size=256, shuffle=True, num_workers=0)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset_test = CelebA('Eval', transforms=transform_test)
    test_label = [get_labels(dataset_test)[x] for x in range(len(get_labels(dataset_test)))]
    test_non_target_1 = list(np.where(np.array(test_label) != target_label_1)[0])
    test_non_target_2 = list(np.where(np.array(test_label) != target_label_2)[0])

    test_non_target_change_image_label_1 = poison_image_and_label_1(original_test, test_non_target_1, target=target_label_1,  transform=None)
    test_non_target_change_image_label_2 = poison_image_and_label_2(original_test, test_non_target_2,  target=target_label_2, transform=None)
    asr_loaders_1 = torch.utils.data.DataLoader(test_non_target_change_image_label_1, batch_size=256, shuffle=True, num_workers=0)
    asr_loaders_2 = torch.utils.data.DataLoader(test_non_target_change_image_label_2, batch_size=256, shuffle=True, num_workers=0)

    print('Poison test dataset for target_1 size is:', len(test_non_target_change_image_label_1))
    print('Poison test dataset for target_2 size is:', len(test_non_target_change_image_label_2))

    clean_test_loader = torch.utils.data.DataLoader(original_test, batch_size=256, shuffle=False, num_workers=0)
    test_target_1 = list(np.where(np.array(test_label) == target_label_1)[0])
    test_target_2 = list(np.where(np.array(test_label) == target_label_2)[0])
    target_test_set_1 = Subset(original_test, test_target_1)
    target_test_set_2 = Subset(original_test, test_target_2)



    target_test_loader_1 = torch.utils.data.DataLoader(target_test_set_1, batch_size=256, shuffle=True, num_workers=0)
    target_test_loader_2 = torch.utils.data.DataLoader(target_test_set_2, batch_size=256, shuffle=True, num_workers=0)

    print('\nTrain start')
    for epoch in range(0, train_epoch):
        test_model.train()
        print('\nEpoch: ' + str(epoch + 1))
        train_accuracy, train_loss = 0.0, 0.0
        total = 0
        for batch_idx, (images, labels) in enumerate(poi_train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = test_model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = out.argmax(axis=1)
            total += labels.size(0)
            train_accuracy += preds.eq(labels).sum().item()
            progress_bar(batch_idx, len(poi_train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * train_accuracy / total, train_accuracy, total))

        attack_accuracy_0, attack_loss_0 = 0.0, 0.0
        attack_sum_0 = 0
        test_model.eval()
        print('Attack start for target_1 Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(asr_loaders_1):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                attack_loss_0 += loss.item()
                preds = out.argmax(axis=1)

                attack_sum_0 += labels.size(0)
                attack_accuracy_0 += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(asr_loaders_1), 'Loss: %.3f | Attack Acc: %.3f%% (%d/%d)'
                             % (attack_loss_0 / (batch_idx + 1), 100. * attack_accuracy_0 / attack_sum_0, attack_accuracy_0, attack_sum_0))

        attack_accuracy_1, attack_loss_1 = 0.0, 0.0
        attack_sum_1 = 0
        print('Attack start for target_2 Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(asr_loaders_2):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                attack_loss_1 += loss.item()
                preds = out.argmax(axis=1)

                attack_sum_1 += labels.size(0)
                attack_accuracy_1 += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(asr_loaders_2), 'Loss: %.3f | Attack Acc: %.3f%% (%d/%d)'
                             % (attack_loss_1 / (batch_idx + 1), 100. * attack_accuracy_1 / attack_sum_1,
                                attack_accuracy_1, attack_sum_1))

        val_accuracy, val_loss = 0.0, 0.0
        sum = 0
        print('Val_Clean start for Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(clean_test_loader):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                val_loss += loss.item()
                preds = out.argmax(axis=1)

                sum += labels.size(0)
                val_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(clean_test_loader), 'Loss: %.3f | Clean Acc: %.3f%% (%d/%d)'
                             % (val_loss / (batch_idx + 1), 100. * val_accuracy / sum, val_accuracy, sum))

        target0_accuracy, target0_loss = 0.0, 0.0
        target0_sum = 0
        print('Val_Target0 start for Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(target_test_loader_1):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                target0_loss += loss.item()
                preds = out.argmax(axis=1)

                target0_sum += labels.size(0)
                target0_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(target_test_loader_1), 'Loss: %.3f | Target Acc: %.3f%% (%d/%d)'
                             % (target0_loss / (batch_idx + 1), 100. * target0_accuracy / target0_sum, target0_accuracy, target0_sum))

        target1_accuracy, target1_loss = 0.0, 0.0
        target1_sum = 0
        print('Val_Target1 start for Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(target_test_loader_2):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                target1_loss += loss.item()
                preds = out.argmax(axis=1)

                target1_sum += labels.size(0)
                target1_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(target_test_loader_2), 'Loss: %.3f | Target Acc: %.3f%% (%d/%d)'
                             % (target1_loss / (batch_idx + 1), 100. * target1_accuracy / target1_sum, target1_accuracy,
                                target1_sum))

if __name__ == '__main__':
    main()

