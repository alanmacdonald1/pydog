import torch
import torchvision


def get_model(device):
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(2048, 2)
    return model.to(device)
