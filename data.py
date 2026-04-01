import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

DATA_DIR = "data/dogs"
BATCH_SIZE = 32


def get_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    display_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    return train_transforms, valid_transforms, display_transforms


def get_dataloaders(train_transforms, valid_transforms, device):
    train_dataset = datasets.ImageFolder(DATA_DIR + "/train", transform=train_transforms)
    valid_dataset = datasets.ImageFolder(DATA_DIR + "/valid", transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    return train_loader, valid_loader, train_dataset
