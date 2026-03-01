from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from utils import *
from util import *


def main():

    device = 'mps'
    train_epoch = 200
    transform = transforms.Compose([
        transforms.Resize((128)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CelebA('Train', transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    test_dataset = CelebA('Eval', transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, drop_last=True)

    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 8)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print('\nTrain start')
    for epoch in range(0, train_epoch):
        model.train()
        print('\nEpoch: ' + str(epoch + 1))
        train_accuracy, train_loss = 0.0, 0.0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = out.argmax(axis=1)
            total += labels.size(0)
            train_accuracy += preds.eq(labels).sum().item()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * train_accuracy / total, train_accuracy, total))

        val_accuracy, val_loss = 0.0, 0.0
        sum = 0
        model.eval()
        print('Validation start for Epoch: ' + str(epoch + 1))
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                loss = criterion(out, labels)
                val_loss += loss.item()
                preds = out.argmax(axis=1)

                sum += labels.size(0)
                val_accuracy += preds.eq(labels).sum().item()
                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (val_loss / (batch_idx + 1), 100. * val_accuracy / sum, val_accuracy, sum))
    # Save the surrogate model
    save_path = './checkpoint/model_pretrain_' + str(train_epoch) + '.pth'
    torch.save(model.state_dict(), save_path)



if __name__ == '__main__':
    main()