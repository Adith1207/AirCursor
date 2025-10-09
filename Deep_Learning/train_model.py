import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# -----------------------------
# GoogLeNet Gesture Trainer
# -----------------------------
class GoogLeNetGestureTrainer:
    def __init__(self, num_classes=2, pretrained=True, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load GoogLeNet (latest API)
        weights = models.GoogLeNet_Weights.DEFAULT if pretrained else None
        self.model = models.googlenet(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)

    def train(self, train_loader, val_loader, epochs=10, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = 100 * correct / total
            val_acc = self.evaluate(val_loader)
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Loss: {total_loss/len(train_loader):.4f} "
                  f"Train Acc: {train_acc:.2f}% "
                  f"Val Acc: {val_acc:.2f}%")

    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return 100 * correct / total

    def save(self, path="googlenet_gesture.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"✅ Model saved at {path}")

# -----------------------------
# Train Script
# -----------------------------
if __name__ == "__main__":
    data_dir = "dataset"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    num_classes = len(train_data.classes)
    trainer = GoogLeNetGestureTrainer(num_classes=num_classes)
    trainer.train(train_loader, val_loader, epochs=10, lr=0.001)
    trainer.save()
