#dataset_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import os
import random
from PIL import Image

# Define paths
data_dir = "data/plant_disease_data/"
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid")
test_dir = os.path.join(data_dir, "test")  # This should contain images directly (no subfolders)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load train and validation datasets using ImageFolder (these have subdirectories)
full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
full_valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)

# Set dataset limits
train_limit = 10000  # Limit the number of training images
valid_limit = 10000  # Limit the number of validation images

# Select random subset indices
train_indices = random.sample(range(len(full_train_dataset)), min(train_limit, len(full_train_dataset)))
valid_indices = random.sample(range(len(full_valid_dataset)), min(valid_limit, len(full_valid_dataset)))

# Create subset datasets
train_dataset = Subset(full_train_dataset, train_indices)
valid_dataset = Subset(full_valid_dataset, valid_indices)

# Custom dataset class for test images (since they are in a single folder)
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB to avoid grayscale issues
        if self.transform:
            image = self.transform(image)
        return image  # No label since test images are unlabeled

# Load test dataset
test_dataset = TestDataset(test_dir, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)  
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Print dataset sizes after applying limits
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(valid_dataset)}")
print(f"Test images: {len(test_dataset)}")
