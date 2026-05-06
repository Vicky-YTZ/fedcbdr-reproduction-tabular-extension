import torch
from torch.utils.data import DataLoader
from model import extract_features


def extract_local_features(model, dataset, device, batch_size=128, extract_fn=extract_features):
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            feats = extract_fn(model, inputs).cpu()
            all_features.append(feats)
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels


