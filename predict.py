#predict.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import PlantDiseaseCNN
from dataset_loader import test_dataset, train_dataset, full_train_dataset  # Ensure you import full_train_dataset

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
num_classes = len(full_train_dataset.classes)  # Use the original full train dataset to get the classes
model = PlantDiseaseCNN(num_classes).to(device)
model.load_state_dict(torch.load("models/plant_disease_model.pth", map_location=device))
model.eval()

# Define image transformations (same as used in training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get class labels
class_labels = full_train_dataset.classes  # Class names from the full training dataset

# Function to predict a single image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_labels[predicted.item()]

    return predicted_class

# Predict images in the test folder
test_images = [os.path.join("data/plant_disease_data/test", img) for img in os.listdir("data/plant_disease_data/test") if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_path in test_images[:10]:  # Predict first 10 images for demonstration
    predicted_label = predict_image(img_path)
    print(f"Image: {os.path.basename(img_path)}, Predicted Disease: {predicted_label}")

