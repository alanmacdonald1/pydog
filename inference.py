import torch
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_PATH = "test_data/img.png"


def predict_and_plot(model, train_dataset, valid_transforms, display_transforms, device):
    image = Image.open(IMAGE_PATH).convert("RGB")
    image_tensor = valid_transforms(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = torch.argmax(probs).item()

    label = train_dataset.classes[pred_idx]
    confidence = probs[pred_idx].item()

    print(f"{label} ({confidence:.2%})")

    plt.imshow(display_transforms(image))
    plt.title(f"{label} ({confidence:.2%})")
    plt.axis("off")
    plt.show()
