from config import DEVICE
from data import get_transforms, get_dataloaders
from model import get_model
from train import train
from evaluate import validate
from inference import predict_and_plot
from test import sleeper
print("pydog.py __name__:", __name__)


def main():

    train_t, valid_t, display_t = get_transforms()
    train_loader, valid_loader, train_dataset = get_dataloaders(train_t, valid_t, DEVICE)
    model = get_model(DEVICE)

    train(model, train_loader, DEVICE)
    validate(model, valid_loader, DEVICE)

    predict_and_plot(model, train_dataset, valid_t, display_t, DEVICE)

if __name__ == "__main__":
    main()
