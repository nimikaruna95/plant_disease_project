#train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import train_loader, valid_loader, full_train_dataset  # Import full dataset
from model import PlantDiseaseCNN

def train():
    # Define device (Force CPU if no CUDA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA not available. Using CPU (Training will be slow)")

    # Initialize model
    num_classes = len(full_train_dataset.classes)  # ✅ Use full_train_dataset to get class count
    model = PlantDiseaseCNN(num_classes).to(device)
    print("✅ Model initialized.")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 10

    print("🔥 Starting training...")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"🟢 Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation step
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"🔵 Validation Accuracy: {val_accuracy:.2f}%")

    # Save trained model
    torch.save(model.state_dict(), "models/plant_disease_model.pth")
    print("✅ Model saved successfully.")

# ✅ Fix multiprocessing issue on Windows
if __name__ == '__main__':
    train()

 