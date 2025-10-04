# src/model.py
import torch
import torch.nn as nn
from torchvision import models




def build_resnet18(num_classes, pretrained=True, freeze_backbone=False):
model = models.resnet18(pretrained=pretrained)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)


if freeze_backbone:
# freeze all except final fc
for name, param in model.named_parameters():
if 'fc' not in name:
param.requires_grad = False
return model