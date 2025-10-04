# src/data_loader.py
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_cifar10_loaders(data_dir, batch_size=64, val_split=0.1, num_workers=2):
train_transform = transforms.Compose([
transforms.RandomResizedCrop(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_transform = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)


val_size = int(len(full_train) * val_split)
train_size = len(full_train) - val_size
train_set, val_set = random_split(full_train, [train_size, val_size])


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


classes = full_train.classes
return train_loader, val_loader, test_loader, classes
