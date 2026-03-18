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
