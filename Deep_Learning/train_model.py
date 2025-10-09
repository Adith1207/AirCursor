import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import os
from tqdm import tqdm


class GoogLeNetArchitecture(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(GoogLeNetArchitecture, self).__init__()

        # ✅ Updated for modern torchvision
        weights = models.GoogLeNet_Weights.DEFAULT if pretrained else None
        self.model = models.googlenet(weights=weights, aux_logits=True)

        # Replace final classifier layers
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Replace auxiliary classifiers if aux_logits=True
        if self.model.aux1:
            aux1_features = self.model.aux1.fc2.in_features
            self.model.aux1.fc2 = nn.Linear(aux1_features, num_classes)

        if self.model.aux2:
            aux2_features = self.model.aux2.fc2.in_features
            self.model.aux2.fc2 = nn.Linear(aux2_features, num_classes)

    def forward(self, x):
        return self.model(x)


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        if isinstance(outputs, tuple):
            main_output, aux1_output, aux2_output = outputs
            loss = (criterion(main_output, labels) +
                    0.3 * criterion(aux1_output, labels) +
                    0.3 * criterion(aux2_output, labels))
            _, preds = torch.max(main_output, 1)
        else:
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item()
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    # === Configuration ===
    data_dir = "dataset"  # Your dataset folder
    batch_size = 16
    num_classes = 5       # Change according to your dataset
    num_epochs = 10
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Data Loading ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # === Model Setup ===
    model = GoogLeNetArchitecture(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # === Training Loop ===
    print("🚀 Starting Training on", device)
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # === Save Model ===
    torch.save(model.state_dict(), "googlenet_final.pth")
    print("\n✅ Model saved as googlenet_final.pth")
