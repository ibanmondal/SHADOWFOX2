import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image, ImageDraw, ImageFont
import random

# CIFAR-10 classes
CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("checkpoints/resnet18_cifar10.pth", map_location="cpu"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-10 test set
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
indices = random.sample(range(len(testset)), 10)  # pick 10 random images

for idx in indices:
    img, label = testset[idx]
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        pred_label = CLASSES[predicted.item()]

    # Save image with predicted label
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.text((5, 5), f"Pred: {pred_label}", fill=(255, 0, 0))
    img_copy.save(f"outputs/pred_{idx}.png")

print("Saved 10 random predictions in outputs/")
