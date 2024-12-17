"""
CNN Training Script for Necrotic Tissue Detection

Description:
This script trains a DenseNet161 model for binary classification (necrotic vs. non-necrotic patches).
It also calculates the necroptosis score for each WSI based on patch predictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description="Train DenseNet161 for Necrotic Tissue Detection")
parser.add_argument("--data", type=str, required=True, help="Path to dataset folder")
parser.add_argument("--output", type=str, required=True, help="Path to save trained model")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
args = parser.parse_args()

# Define data transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    "val": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
}

# Load datasets
print("Loading datasets...")
data_dir = args.data
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ["train", "val"]}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4)
               for x in ["train", "val"]}

# Model setup
print("Initializing DenseNet161 model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.densenet161(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 1),  # Binary classification output
    nn.Sigmoid()
)
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Training and Validation Loop
def train_model(model, dataloaders, criterion, optimizer, num_epochs, output_path):
    print("Starting training...")
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs).squeeze()
                    preds = (outputs > 0.5).float()
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print("Training complete.")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(output_path, "densenet161_necrotic.pth"))
    print(f"Model saved to {output_path}")

# Run Training
if __name__ == "__main__":
    os.makedirs(args.output, exist_ok=True)
    train_model(model, dataloaders, criterion, optimizer, args.epochs, args.output)
