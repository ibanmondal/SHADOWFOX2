import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image
import sys

# CIFAR-10 classes
CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

def load_model(checkpoint_path="checkpoints/resnet18_cifar10.pth"):
    # Load ResNet18 with the correct architecture
    model = models.resnet18(weights=None)  # no pretrained, use your trained weights
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image_path, checkpoint_path="checkpoints/resnet18_cifar10.pth"):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension

    # Load model
    model = load_model(checkpoint_path)

    # Run prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = CLASSES[predicted.item()]

    print(f"Predicted Class: {label}")
    return label

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
    else:
        image_path = sys.argv[1]
        predict(image_path)
