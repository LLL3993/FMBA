import os
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import fft
from torchvision import datasets, transforms
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score
import scipy.stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_cifar10_official(root_dir):
    train_x, train_y = [], []
    for i in range(1, 6):
        with open(os.path.join(root_dir, f'data_batch_{i}'), 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            train_x.append(d[b'data'])
            train_y += d[b'labels']
    train_x = np.vstack(train_x).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_y = np.array(train_y)

    with open(os.path.join(root_dir, 'test_batch'), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    test_x = d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_y = np.array(d[b'labels'])
    return (train_x, train_y), (test_x, test_y)

data_root = '/data1/liuyiyang/safe-work/Muti-Trigger/dataset/CIFAR10/cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_cifar10_official(data_root)
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32') / 255.

trigger0 = torch.load('/data1/liuyiyang/safe-work/Muti-Trigger/Muti-Trigger For Clean-Label Attack/CIFAR10/FMBA/trigger/target0.npy')
trigger1 = torch.load('/data1/liuyiyang/safe-work/Muti-Trigger/Muti-Trigger For Clean-Label Attack/CIFAR10/FMBA/trigger/target1.npy')

def fussion_noise(noise_1, noise_2, alpha):
    fft_image1 = fft.fft2(noise_1)
    fft_image2 = fft.fft2(noise_2)
    blended_fft = alpha * fft_image1 + (1 - alpha) * fft_image2
    fussion_trigger = fft.ifft2(blended_fft)
    fussion_trigger = torch.abs(fussion_trigger)
    return fussion_trigger

alpha_attack = 0.9
trigger_attack = fussion_noise(trigger0, trigger1, alpha_attack)
trigger_attack = trigger_attack + 3 * trigger0
# trigger_attack = trigger1
trigger_attack = np.transpose(trigger_attack.numpy(), (0, 2, 3, 1))

model=torch.load('/data1/liuyiyang/safe-work/Muti-Trigger/Muti-Trigger For Clean-Label Attack/CIFAR10/FMBA/checkpoint/backdoor_model_27.pth', 
                weights_only=False, map_location=device)

model = model.to(device).eval()

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
clean_set = datasets.CIFAR10(root='/data1/liuyiyang/safe-work/Muti-Trigger/dataset/CIFAR10',
                             train=False, download=False, transform=transform_val)
clean_loader = DataLoader(clean_set, batch_size=256, shuffle=False)
all_pred, all_true = [], []
with torch.no_grad():
    for im, lb in clean_loader:
        im = im.to(device)
        all_pred.append(model(im).argmax(1).cpu())
        all_true.append(lb)
acc_clean = accuracy_score(torch.cat(all_true), torch.cat(all_pred))
print(f'[INFO] clean accuracy = {acc_clean*100:.2f}%')

def superimpose(background, overlay):
    added = cv2.addWeighted(background, 1, overlay, 1, 0)
    return added.reshape(32, 32, 3)

def entropyCal(background, n, model, candidate_pool):
    indices = np.random.choice(len(candidate_pool), n, replace=False)
    batch = []
    for i in indices:
        fused = superimpose(background, candidate_pool[i])
        batch.append(transform_val(fused.astype('float32')))
    batch = torch.stack(batch).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(batch), dim=1).cpu().numpy()
    EntropySum = -np.nansum(prob * np.log2(prob + 1e-12))
    # print(prob)
    # print(EntropySum)
    return EntropySum / n

CANDIDATE_OFFSET = 30000
CANDIDATE_NUM    = 5000
candidate_pool = x_train[CANDIDATE_OFFSET:CANDIDATE_OFFSET+CANDIDATE_NUM]

# CLEAN_TEST_NUM = 500
CLEAN_TEST_NUM = 500
clean_test_pool = x_test[:CLEAN_TEST_NUM]

# POISON_TEST_NUM = 500
POISON_TEST_NUM = 500
poison_test_pool = []
for i in range(CLEAN_TEST_NUM, CLEAN_TEST_NUM+POISON_TEST_NUM):
    img = x_test[i].copy()
    img = np.clip(img + trigger_attack[0], 0, 1)
    poison_test_pool.append(img)

# N_SAMPLE = 64
N_SAMPLE = 64
np.random.seed(0)
entropy_clean = [entropyCal(clean_test_pool[i], N_SAMPLE, model, candidate_pool)
                 for i in range(CLEAN_TEST_NUM)]
entropy_poison = [entropyCal(poison_test_pool[i], N_SAMPLE, model, candidate_pool)
                  for i in range(POISON_TEST_NUM)]

mu, sigma = scipy.stats.norm.fit(entropy_clean)
threshold = scipy.stats.norm.ppf(0.01, loc=mu, scale=sigma)
FAR = sum(e > threshold for e in entropy_poison) / POISON_TEST_NUM * 100
print(f'[STRIP] threshold @1%FRR = {threshold:.4f}')
print(f'[STRIP] FAR (poison detected as clean) = {FAR:.2f}%')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(entropy_clean, bins=50, alpha=0.7, label='Clean Samples', density=True, color='blue')
plt.hist(entropy_poison, bins=50, alpha=0.7, label='Poison Samples', density=True, color='red')

plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold (1% FRR)')

plt.xlabel('Entropy', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Entropy Distribution', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.text(0.02, 0.95, f'Clean Mean: {np.mean(entropy_clean):.4f}\nPoison Mean: {np.mean(entropy_poison):.4f}\nFAR: {FAR:.2f}%', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('entropy_distribution.png', dpi=300)