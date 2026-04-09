import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=10):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def extract_features(model, x):
    """
    提取 ResNet18 倒数第二层特征
    输出 shape: [batch_size, feature_dim]
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


import torch
import torch.nn as nn

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

def extract_features(model, x):
    model.eval()
    with torch.no_grad():
        features = model.feature_extractor(x)
    return features
