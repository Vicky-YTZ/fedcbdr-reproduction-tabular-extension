import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=10):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def extract_features(model, x):
    """
    Extract features from the penultimate layer of ResNet18.
    Output shape: [batch_size, feature_dim]
    """
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    return x
class TabularMLP(nn.Module):
    def __init__(self, input_dim, feature_dim=128, num_classes=10):
        super(TabularMLP, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.fc(features)
        return logits

def get_tabular_model(input_dim, num_classes=10):
    return TabularMLP(input_dim=input_dim, feature_dim=128, num_classes=num_classes)

def extract_tabular_features(model, x):
    model.eval()
    with torch.no_grad():
        features = model.feature_extractor(x)
    return features


def expand_model_classifier(model, new_num_classes, device):
    """
    Expand the final Linear layer for a new Task while preserving the old weights.
    """
    old_fc = model.fc
    old_out_features = old_fc.out_features
    in_features = old_fc.in_features
    
    # If the current number of classes is sufficient, do nothing
    if old_out_features >= new_num_classes:
        return model
        
    # Create a new, larger Linear layer
    new_fc = nn.Linear(in_features, new_num_classes).to(device)
    
    # Copy old weights to the beginning of the new layer
    new_fc.weight.data[:old_out_features] = old_fc.weight.data
    new_fc.bias.data[:old_out_features] = old_fc.bias.data
    
    model.fc = new_fc
    return model
