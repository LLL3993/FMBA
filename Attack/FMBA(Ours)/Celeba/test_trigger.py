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


def main():
    # # 固定训练结果
    # random_seed = 0
    # np.random.seed(random_seed)
    # random.seed(random_seed)
    # torch.manual_seed(random_seed)

    noise_1 = torch.load("trigger/target0.npy")
    noise_2 = torch.load("trigger/target2.npy")
    noise_3 = torch.load("trigger/target3.npy")
    noise_4 = torch.load("trigger/target1.npy")
    noise_5 = torch.load("trigger/target4.npy")
    noise_6 = torch.load("trigger/target5.npy")
    noise_7 = torch.load("trigger/target6.npy")
    noise_8 = torch.load("trigger/target7.npy")

    fussion_trigger_1 = fussion_noise(noise_1, noise_2, 0.5)
    fussion_trigger_2 = fussion_noise(fussion_trigger_1, noise_3, 0.5)
    fussion_trigger_3 = fussion_noise(fussion_trigger_2, noise_4, 0.5)
    fussion_trigger_4 = fussion_noise(fussion_trigger_3, noise_5, 0.5)
    fussion_trigger_5 = fussion_noise(fussion_trigger_4, noise_6, 0.5)
    fussion_trigger_6 = fussion_noise(fussion_trigger_5, noise_7, 0.5)
    fussion_trigger = fussion_noise(fussion_trigger_6, noise_8, 0.5)


    device = 'mps'

    poi_num = 200

    test_model = resnet18(weights=None)
    test_model.fc = nn.Linear(test_model.fc.in_features, 8)
    test_model = test_model.to(device)

    train_epoch = 100

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(test_model.parameters(), lr=0.001)

    transform_tensor = transforms.Compose([
        transforms.Resize((128)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    original_train = CelebA('Train', transforms=transform_tensor)
    original_test = CelebA('Eval', transforms=transform_tensor)

    transform_after_tensor = transforms.Compose([

    ])

    target_label_1 = 0
    target_label_2 = 2
    target_label_3 = 3
    target_label_4 = 1
    target_label_5 = 4
    target_label_6 = 5
    target_label_7 = 6
    target_label_8 = 7

    transform_train = transforms.Compose([
        transforms.Resize((128)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_train = CelebA('Train', transforms=transform_train)
    train_label = [get_labels(dataset_train)[x] for x in range(len(get_labels(dataset_train)))]
    train_target_list_1 = list(np.where(np.array(train_label) == target_label_1)[0])
    train_target_list_2 = list(np.where(np.array(train_label) == target_label_2)[0])
    train_target_list_3 = list(np.where(np.array(train_label) == target_label_3)[0])
    train_target_list_4 = list(np.where(np.array(train_label) == target_label_4)[0])
    train_target_list_5 = list(np.where(np.array(train_label) == target_label_5)[0])
    train_target_list_6 = list(np.where(np.array(train_label) == target_label_6)[0])
    train_target_list_7 = list(np.where(np.array(train_label) == target_label_7)[0])
    train_target_list_8 = list(np.where(np.array(train_label) == target_label_8)[0])
    # train_target_list_final = train_target_list_1 + train_target_list_2 + target_label_3
    random_poison_idx_1 = random.sample(train_target_list_1, poi_num)
    random_poison_idx_2 = random.sample(train_target_list_2, poi_num)
    random_poison_idx_3 = random.sample(train_target_list_3, poi_num)
    random_poison_idx_4 = random.sample(train_target_list_4, poi_num)
    random_poison_idx_5 = random.sample(train_target_list_5, poi_num)
    random_poison_idx_6 = random.sample(train_target_list_6, poi_num)
    random_poison_idx_7 = random.sample(train_target_list_7, poi_num)
    random_poison_idx_8 = random.sample(train_target_list_8, poi_num)

    random_poison_idx_final = random_poison_idx_1 + random_poison_idx_2 + random_poison_idx_3 + random_poison_idx_4 + random_poison_idx_5 + random_poison_idx_6 + random_poison_idx_7 + random_poison_idx_8

    # print(random_poison_idx)
    poison_train_target = poison_image_without_label(original_train, random_poison_idx_final, noise_1, noise_2, noise_3, noise_4, noise_5, noise_6, noise_7, noise_8, target_label_1, target_label_2, target_label_3, target_label_4, target_label_5, target_label_6, target_label_7, target_label_8, transform_after_tensor)
    print('Train dataset size is:', len(poison_train_target), " Poison numbers is:", len(random_poison_idx_final))

    poi_train_loader = DataLoader(poison_train_target, batch_size=256, shuffle=True, num_workers=0)

    transform_test = transforms.Compose([
        transforms.Resize((128)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_test = CelebA('Eval', transforms=transform_test)
    test_label = [get_labels(dataset_test)[x] for x in range(len(get_labels(dataset_test)))]
    test_non_target_1 = list(np.where(np.array(test_label) != target_label_1)[0])
    test_non_target_2 = list(np.where(np.array(test_label) != target_label_2)[0])
    test_non_target_3 = list(np.where(np.array(test_label) != target_label_3)[0])
    test_non_target_4 = list(np.where(np.array(test_label) != target_label_4)[0])
    test_non_target_5 = list(np.where(np.array(test_label) != target_label_5)[0])
    test_non_target_6 = list(np.where(np.array(test_label) != target_label_6)[0])
    test_non_target_7 = list(np.where(np.array(test_label) != target_label_7)[0])
    test_non_target_8 = list(np.where(np.array(test_label) != target_label_8)[0])

    test_non_target_change_image_label_1 = poison_image_and_label(original_test, test_non_target_1, trigger=fussion_trigger.cpu(), noise=noise_1.cpu(), target=target_label_1, amplify_multi=3, transform=None)
    test_non_target_change_image_label_2 = poison_image_and_label(original_test, test_non_target_2, trigger=fussion_trigger.cpu(), noise=noise_2.cpu(), target=target_label_2, amplify_multi=3, transform=None)
    test_non_target_change_image_label_3 = poison_image_and_label(original_test, test_non_target_3, trigger=fussion_trigger.cpu(), noise=noise_3.cpu(), target=target_label_3, amplify_multi=3, transform=None)
    test_non_target_change_image_label_4 = poison_image_and_label(original_test, test_non_target_4, trigger=fussion_trigger.cpu(), noise=noise_4.cpu(), target=target_label_4, amplify_multi=3, transform=None)
    test_non_target_change_image_label_5 = poison_image_and_label(original_test, test_non_target_5, trigger=fussion_trigger.cpu(), noise=noise_5.cpu(), target=target_label_5, amplify_multi=3, transform=None)
    test_non_target_change_image_label_6 = poison_image_and_label(original_test, test_non_target_6, trigger=fussion_trigger.cpu(), noise=noise_6.cpu(), target=target_label_6, amplify_multi=3, transform=None)
    test_non_target_change_image_label_7 = poison_image_and_label(original_test, test_non_target_7, trigger=fussion_trigger.cpu(), noise=noise_7.cpu(), target=target_label_7, amplify_multi=3, transform=None)
    test_non_target_change_image_label_8 = poison_image_and_label(original_test, test_non_target_8, trigger=fussion_trigger.cpu(), noise=noise_8.cpu(), target=target_label_8, amplify_multi=3, transform=None)
    asr_loaders_1 = torch.utils.data.DataLoader(test_non_target_change_image_label_1, batch_size=256, shuffle=True, num_workers=0)
    asr_loaders_2 = torch.utils.data.DataLoader(test_non_target_change_image_label_2, batch_size=256, shuffle=True, num_workers=0)
    asr_loaders_3 = torch.utils.data.DataLoader(test_non_target_change_image_label_3, batch_size=256, shuffle=True, num_workers=0)
    asr_loaders_4 = torch.utils.data.DataLoader(test_non_target_change_image_label_4, batch_size=256, shuffle=True, num_workers=0)
    asr_loaders_5 = torch.utils.data.DataLoader(test_non_target_change_image_label_5, batch_size=256, shuffle=True, num_workers=0)
    asr_loaders_6 = torch.utils.data.DataLoader(test_non_target_change_image_label_6, batch_size=256, shuffle=True, num_workers=0)
    asr_loaders_7 = torch.utils.data.DataLoader(test_non_target_change_image_label_7, batch_size=256, shuffle=True, num_workers=0)
    asr_loaders_8 = torch.utils.data.DataLoader(test_non_target_change_image_label_8, batch_size=256, shuffle=True, num_workers=0)

    print('Poison test dataset for target_1 size is:', len(test_non_target_change_image_label_1))
    print('Poison test dataset for target_2 size is:', len(test_non_target_change_image_label_2))
    print('Poison test dataset for target_3 size is:', len(test_non_target_change_image_label_3))
    print('Poison test dataset for target_4 size is:', len(test_non_target_change_image_label_4))
    print('Poison test dataset for target_5 size is:', len(test_non_target_change_image_label_5))
    print('Poison test dataset for target_6 size is:', len(test_non_target_change_image_label_6))
    print('Poison test dataset for target_7 size is:', len(test_non_target_change_image_label_7))
    print('Poison test dataset for target_8 size is:', len(test_non_target_change_image_label_8))

    clean_test_loader = torch.utils.data.DataLoader(original_test, batch_size=256, shuffle=False, num_workers=0)
    test_target_1 = list(np.where(np.array(test_label) == target_label_1)[0])
    test_target_2 = list(np.where(np.array(test_label) == target_label_2)[0])
    test_target_3 = list(np.where(np.array(test_label) == target_label_3)[0])
    test_target_4 = list(np.where(np.array(test_label) == target_label_4)[0])
    test_target_5 = list(np.where(np.array(test_label) == target_label_5)[0])
    test_target_6 = list(np.where(np.array(test_label) == target_label_6)[0])
    test_target_7 = list(np.where(np.array(test_label) == target_label_7)[0])
    test_target_8 = list(np.where(np.array(test_label) == target_label_8)[0])
    target_test_set_1 = Subset(original_test, test_target_1)
    target_test_set_2 = Subset(original_test, test_target_2)
    target_test_set_3 = Subset(original_test, test_target_3)
    target_test_set_4 = Subset(original_test, test_target_4)
    target_test_set_5 = Subset(original_test, test_target_5)
    target_test_set_6 = Subset(original_test, test_target_6)
    target_test_set_7 = Subset(original_test, test_target_7)
    target_test_set_8 = Subset(original_test, test_target_8)

    target_test_loader_1 = torch.utils.data.DataLoader(target_test_set_1, batch_size=256, shuffle=True, num_workers=0)
    target_test_loader_2 = torch.utils.data.DataLoader(target_test_set_2, batch_size=256, shuffle=True, num_workers=0)
    target_test_loader_3 = torch.utils.data.DataLoader(target_test_set_3, batch_size=256, shuffle=True, num_workers=0)
    target_test_loader_4 = torch.utils.data.DataLoader(target_test_set_4, batch_size=256, shuffle=True, num_workers=0)
    target_test_loader_5 = torch.utils.data.DataLoader(target_test_set_5, batch_size=256, shuffle=True, num_workers=0)
    target_test_loader_6 = torch.utils.data.DataLoader(target_test_set_6, batch_size=256, shuffle=True, num_workers=0)
    target_test_loader_7 = torch.utils.data.DataLoader(target_test_set_7, batch_size=256, shuffle=True, num_workers=0)
    target_test_loader_8 = torch.utils.data.DataLoader(target_test_set_8, batch_size=256, shuffle=True, num_workers=0)

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

        attack_accuracy_2, attack_loss_2 = 0.0, 0.0
        attack_sum_2 = 0
        print('Attack start for target_3 Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(asr_loaders_3):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                attack_loss_2 += loss.item()
                preds = out.argmax(axis=1)

                attack_sum_2 += labels.size(0)
                attack_accuracy_2 += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(asr_loaders_3), 'Loss: %.3f | Attack Acc: %.3f%% (%d/%d)'
                             % (attack_loss_2 / (batch_idx + 1), 100. * attack_accuracy_2 / attack_sum_2,
                                attack_accuracy_2, attack_sum_2))

        attack_accuracy_3, attack_loss_3 = 0.0, 0.0
        attack_sum_3 = 0
        print('Attack start for target_4 Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(asr_loaders_4):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                attack_loss_3 += loss.item()
                preds = out.argmax(axis=1)

                attack_sum_3 += labels.size(0)
                attack_accuracy_3 += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(asr_loaders_4), 'Loss: %.3f | Attack Acc: %.3f%% (%d/%d)'
                             % (attack_loss_3 / (batch_idx + 1), 100. * attack_accuracy_3 / attack_sum_3,
                                attack_accuracy_3, attack_sum_3))
        attack_accuracy_4, attack_loss_4 = 0.0, 0.0
        attack_sum_4 = 0
        print('Attack start for target_5 Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(asr_loaders_5):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                attack_loss_4 += loss.item()
                preds = out.argmax(axis=1)

                attack_sum_4 += labels.size(0)
                attack_accuracy_4 += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(asr_loaders_5), 'Loss: %.3f | Attack Acc: %.3f%% (%d/%d)'
                             % (attack_loss_4 / (batch_idx + 1), 100. * attack_accuracy_4 / attack_sum_4,
                                attack_accuracy_4, attack_sum_4))
        attack_accuracy_5, attack_loss_5 = 0.0, 0.0
        attack_sum_5 = 0
        print('Attack start for target_6 Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(asr_loaders_6):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                attack_loss_5 += loss.item()
                preds = out.argmax(axis=1)

                attack_sum_5 += labels.size(0)
                attack_accuracy_5 += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(asr_loaders_6), 'Loss: %.3f | Attack Acc: %.3f%% (%d/%d)'
                             % (attack_loss_5 / (batch_idx + 1), 100. * attack_accuracy_5 / attack_sum_5,
                                attack_accuracy_5, attack_sum_5))
        attack_accuracy_6, attack_loss_6 = 0.0, 0.0
        attack_sum_6 = 0
        print('Attack start for target_7 Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(asr_loaders_7):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                attack_loss_6 += loss.item()
                preds = out.argmax(axis=1)

                attack_sum_6 += labels.size(0)
                attack_accuracy_6 += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(asr_loaders_7), 'Loss: %.3f | Attack Acc: %.3f%% (%d/%d)'
                             % (attack_loss_6 / (batch_idx + 1), 100. * attack_accuracy_6 / attack_sum_6,
                                attack_accuracy_6, attack_sum_6))
        attack_accuracy_7, attack_loss_7 = 0.0, 0.0
        attack_sum_7 = 0
        print('Attack start for target_8 Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(asr_loaders_8):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                attack_loss_7 += loss.item()
                preds = out.argmax(axis=1)

                attack_sum_7 += labels.size(0)
                attack_accuracy_7 += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(asr_loaders_8), 'Loss: %.3f | Attack Acc: %.3f%% (%d/%d)'
                             % (attack_loss_7 / (batch_idx + 1), 100. * attack_accuracy_7 / attack_sum_7,
                                attack_accuracy_7, attack_sum_7))

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

        target2_accuracy, target2_loss = 0.0, 0.0
        target2_sum = 0
        print('Val_Target2 start for Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(target_test_loader_3):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                target2_loss += loss.item()
                preds = out.argmax(axis=1)

                target2_sum += labels.size(0)
                target2_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(target_test_loader_3), 'Loss: %.3f | Target Acc: %.3f%% (%d/%d)'
                             % (target2_loss / (batch_idx + 1), 100. * target2_accuracy / target2_sum, target2_accuracy,
                                target2_sum))

        target3_accuracy, target3_loss = 0.0, 0.0
        target3_sum = 0
        print('Val_Target3 start for Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(target_test_loader_4):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                target3_loss += loss.item()
                preds = out.argmax(axis=1)

                target3_sum += labels.size(0)
                target3_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(target_test_loader_4), 'Loss: %.3f | Target Acc: %.3f%% (%d/%d)'
                             % (target3_loss / (batch_idx + 1), 100. * target3_accuracy / target3_sum, target3_accuracy,
                                target3_sum))

        target4_accuracy, target4_loss = 0.0, 0.0
        target4_sum = 0
        print('Val_Target4 start for Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(target_test_loader_5):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                target4_loss += loss.item()
                preds = out.argmax(axis=1)

                target4_sum += labels.size(0)
                target4_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(target_test_loader_5), 'Loss: %.3f | Target Acc: %.3f%% (%d/%d)'
                             % (target4_loss / (batch_idx + 1), 100. * target4_accuracy / target4_sum, target4_accuracy,
                                target4_sum))
        target5_accuracy, target5_loss = 0.0, 0.0
        target5_sum = 0
        print('Val_Target5 start for Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(target_test_loader_6):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                target5_loss += loss.item()
                preds = out.argmax(axis=1)

                target5_sum += labels.size(0)
                target5_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(target_test_loader_6), 'Loss: %.3f | Target Acc: %.3f%% (%d/%d)'
                             % (target5_loss / (batch_idx + 1), 100. * target5_accuracy / target5_sum, target5_accuracy,
                                target5_sum))

        target6_accuracy, target6_loss = 0.0, 0.0
        target6_sum = 0
        print('Val_Target6 start for Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(target_test_loader_7):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                target6_loss += loss.item()
                preds = out.argmax(axis=1)

                target6_sum += labels.size(0)
                target6_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(target_test_loader_7), 'Loss: %.3f | Target Acc: %.3f%% (%d/%d)'
                             % (target6_loss / (batch_idx + 1), 100. * target6_accuracy / target6_sum, target6_accuracy,
                                target6_sum))

        target7_accuracy, target7_loss = 0.0, 0.0
        target7_sum = 0
        print('Val_Target7 start for Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(target_test_loader_8):
                images, labels = images.to(device), labels.to(device)
                out = test_model(images)
                loss = criterion(out, labels)
                target7_loss += loss.item()
                preds = out.argmax(axis=1)

                target7_sum += labels.size(0)
                target7_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(target_test_loader_8), 'Loss: %.3f | Target Acc: %.3f%% (%d/%d)'
                             % (target7_loss / (batch_idx + 1), 100. * target7_accuracy / target7_sum, target7_accuracy,
                                target7_sum))

        target0_accuracy = 100. * attack_accuracy_0 / attack_sum_0
        target1_accuracy = 100. * attack_accuracy_1 / attack_sum_1
        target2_accuracy = 100. * attack_accuracy_2 / attack_sum_2
        target3_accuracy = 100. * attack_accuracy_3 / attack_sum_3
        target4_accuracy = 100. * attack_accuracy_4 / attack_sum_4
        target5_accuracy = 100. * attack_accuracy_5 / attack_sum_5
        target6_accuracy = 100. * attack_accuracy_6 / attack_sum_6
        target7_accuracy = 100. * attack_accuracy_7 / attack_sum_7

        # save_path = './checkpoint1/backdoor_model_' + str(epoch) + '.pth'
        # if target0_accuracy > 90.0 and target1_accuracy > 90.0:
        # torch.save(test_model, save_path)

if __name__ == '__main__':
    main()

