import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import PlantDiseaseCNN
from dataset_loader import full_train_dataset  # Import the full train dataset

# Set Streamlit page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Define device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
num_classes = len(full_train_dataset.classes)  # Use full_train_dataset to get the classes
model = PlantDiseaseCNN(num_classes).to(device)
model.load_state_dict(torch.load("models/plant_disease_model.pth", map_location=device))
model.eval()

# Define image transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get class labels from the full dataset
class_labels = full_train_dataset.classes  # List of class names

# Function to predict disease from an image
def predict_disease(image):
    image = transform(image).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_labels[predicted.item()]
    return predicted_class

# Streamlit UI
st.title("🌱 Plant Disease Detection App")
st.write("Upload an image of a plant leaf, and the model will predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict disease when the user clicks the button
    if st.button("Predict Disease"):
        with st.spinner("Analyzing Image..."):
            predicted_label = predict_disease(image)
        st.success(f"🔍 Predicted Disease: **{predicted_label}**")

# Footer
st.write("---")
st.write("📌 Developed for **Plant Disease Detection using Deep Learning**")
