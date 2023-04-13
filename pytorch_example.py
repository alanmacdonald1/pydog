#In this code, we use a pre-trained ResNet-50 model from the torchvision library to classify images of dogs


import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt


from PIL import Image

# Define data directory
data_dir = "data/dogs"

# Define transforms for training data
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define transforms for validation data
valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
valid_dataset = datasets.ImageFolder(data_dir + "/valid", transform=valid_transforms)

# Define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

# Define the model
#  loads a pre-trained ResNet-50 model from the torchvision library, which has been trained on the ImageNet dataset
model = torchvision.models.resnet50(pretrained=True)

# replaces the last fully connected layer of the ResNet-50 model with a new one that has 2 output units, corresponding to the number of classes in the dog dataset.
model.fc = torch.nn.Linear(2048, 2)



# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("Epoch: {}, Loss: {:.4f}".format(epoch+1, loss.item()))

# Test the model on validation data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in valid_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy on validation data: {:.2f}%".format(100 * correct / total))



# Load the image and apply the same transformations as for the validation set
image_path = "newdata/img.png"
image = Image.open(image_path)
image_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_tensor = image_transforms(image)

# Convert the tensor back to an image and show it
image_to_show = transforms.ToPILImage()(image_tensor.squeeze())
plt.imshow(image_to_show)
plt.show()


image_tensor = image_tensor.unsqueeze(0)  # add batch dimension

# Make a prediction on the image
model.eval()
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class_index = torch.argmax(probabilities).item()

# Print the predicted class label
class_labels = train_dataset.classes
predicted_class_label = class_labels[predicted_class_index]
print("Predicted class label: {}".format(predicted_class_label))
