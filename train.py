import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F

def train_one_epoch(model, dataloader, device, replay_dataset=None, batch_size=64):
    model.train()

    # If there is replay data, concatenate the current task data with the replay data
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
    def __init__(self, num_old_classes, tau_old=0.5, tau_new=1.5, w_old=1.5, w_new=1.0):
        super(TTSLoss, self).__init__()
        self.num_old_classes = num_old_classes
        self.tau_old = tau_old
        self.tau_new = tau_new
        self.w_old = w_old
        self.w_new = w_new

    def forward(self, logits, labels):
        device = logits.device
        
        # --- Step 1: Scale temperatures by Column (Class) ---
        old_logits = logits[:, :self.num_old_classes] / self.tau_old
        new_logits = logits[:, self.num_old_classes:] / self.tau_new
        scaled_logits = torch.cat([old_logits, new_logits], dim=1)

        # --- Step 2: Scale temperatures by Row (Sample) ---
        is_old_sample = labels < self.num_old_classes
        sample_temp = torch.where(is_old_sample, self.tau_old, self.tau_new).to(device)
        sample_temp = sample_temp.view(-1, 1)
        
        # Second scaling (THE AUTHOR'S TRICK TO PREVENT GRADIENT EXPLOSION)
        scaled_logits = scaled_logits / sample_temp  

        # --- Step 3: Multiply loss weights ---
        weights = torch.where(is_old_sample, self.w_old, self.w_new).to(device)
        losses = F.cross_entropy(scaled_logits, labels, reduction='none')
        
        return (losses * weights).mean()


def train_one_epoch_tts(model, dataloader, device, replay_dataset=None, batch_size=64,
                        task_id=0, num_old_classes=0, num_total_seen_classes=10, use_tts=True, lr=0.01):
    model.train()
    
    if replay_dataset is not None and len(replay_dataset) > 0:
        combined_dataset = ConcatDataset([dataloader.dataset, replay_dataset])
        train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    else:
        train_loader = dataloader

    # 2. OPTIMIZER AS CONFIGURED IN THE PAPER
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    
    # 3. INITIALIZE LOSS FUNCTION (This is where TTSLoss is called!)
    if use_tts and task_id > 0:
        criterion = TTSLoss(num_old_classes, tau_old=0.5, tau_new=1.5, w_old=5.0, w_new=1.0)
    else:
        criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0

    # 4. TRAINING LOOP
    for inputs, labels in train_loader:
        if inputs.size(0) == 1:
            continue 

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        # CALCULATE COMPACT LOSS (Using the initialized class above)
        loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if total == 0:
        return 0.0

    return 100.0 * correct / total