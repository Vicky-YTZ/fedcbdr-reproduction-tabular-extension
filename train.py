import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F

def train_one_epoch(model, dataloader, device, replay_dataset=None, batch_size=64):
    model.train()

    # 如果有 replay 数据，就把当前 task 数据和 replay 数据拼起来
    if replay_dataset is not None:
        combined_dataset = ConcatDataset([dataloader.dataset, replay_dataset])
        dataloader = DataLoader(
            combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    total = 0
    correct = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    return acc

class TTSLoss(nn.Module):
    """
    Task-aware Temperature Scaling (TTS) Loss from FedCBDR.
    """
    def __init__(self, num_old_classes, num_total_seen_classes, tau_old=0.9, tau_new=1.1, w_old=1.1, w_new=0.9):
        super(TTSLoss, self).__init__()
        self.num_old_classes = num_old_classes
        self.num_total_seen_classes = num_total_seen_classes
        
        self.tau_old = tau_old
        self.tau_new = tau_new
        self.w_old = w_old
        self.w_new = w_new

    def forward(self, logits, targets):
        # Drop unseen future classes
        logits = logits[:, :self.num_total_seen_classes]
        
        # Split logits and apply temperature scaling using ONLY seen classes
        logits_old = logits[:, :self.num_old_classes] / self.tau_old
        logits_new = logits[:, self.num_old_classes:] / self.tau_new
        
        scaled_logits = torch.cat([logits_old, logits_new], dim=1)
        ce_loss = F.cross_entropy(scaled_logits, targets, reduction='none')

        weights = torch.where(
            targets < self.num_old_classes,
            torch.tensor(self.w_old, dtype=logits.dtype, device=logits.device),
            torch.tensor(self.w_new, dtype=logits.dtype, device=logits.device)
        )

        return (ce_loss * weights).mean()


def train_one_epoch_tts(model, dataloader, device, replay_dataset=None, batch_size=64,
                        task_id=0, num_old_classes=0, num_total_seen_classes=10):
    model.train()
    if replay_dataset is not None:
        current_len = len(dataloader.dataset)
        replay_len = len(replay_dataset)
        
        num_repeats = max(1, (current_len // replay_len) // 2) 
        
        repeated_replay = ConcatDataset([replay_dataset] * num_repeats)
        combined_dataset = ConcatDataset([dataloader.dataset, repeated_replay])
        
        dataloader = DataLoader(
            combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

    optimizer = optim.SGD(model.parameters(), lr=0.01*(0.5 ** task_id),weight_decay=5e-4)

    # Initialize Criterion
    if task_id == 0:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = TTSLoss(
            num_old_classes=num_old_classes,
            num_total_seen_classes=num_total_seen_classes, # Pass new Param
            tau_old=0.8, tau_new=1.2, w_old=2.0, w_new=0.5 # Tuned stronger parameters for minority class
        )

    total = 0
    correct = 0

    for inputs, labels in dataloader:
        if inputs.size(0) <= 1:
            continue
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        # IMPORTANT MASKING FOR TASK 0 AS WELL
        if task_id == 0:
            outputs = outputs[:, :num_total_seen_classes]
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return 100 * correct / total