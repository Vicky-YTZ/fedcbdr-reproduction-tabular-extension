import torch


def evaluate(model, dataloader, device, num_old_classes=0, use_tts=False, tau_old=0.5, tau_new=1.5):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            if use_tts and num_old_classes > 0:
                logits_old = outputs[:, :num_old_classes] / tau_old
                logits_new = outputs[:, num_old_classes:] / tau_new
                outputs = torch.cat([logits_old, logits_new], dim=1)

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    return acc