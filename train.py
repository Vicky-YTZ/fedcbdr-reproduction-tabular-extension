import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader


def train_one_epoch(model, dataloader, device, replay_dataset=None, batch_size=64):
    model.train()

    # 如果有 replay 数据，就把当前 task 数据和 replay 数据拼起来
    if replay_dataset is not None:
        combined_dataset = ConcatDataset([dataloader.dataset, replay_dataset])
        dataloader = DataLoader(
            combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    total = 0
    correct = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    return acc
