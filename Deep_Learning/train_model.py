import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image


# -----------------------------
# Class 1: GoogLeNet Architecture
# -----------------------------
class GoogLeNetArchitecture:
    def __init__(self, num_classes=10, pretrained=True, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load GoogLeNet
        self.model = models.googlenet(pretrained=pretrained)
        
        # Modify the last layer for gesture classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        self.model = self.model.to(self.device)
        
        # Preprocessing for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    # ---------- Inference ----------
    def invoke(self, img: Image.Image):
        self.model.eval()
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
        return outputs

    # ---------- Training ----------
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                    epochs=5, lr=0.001):
        """
        Simple training loop for GoogLeNet
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            val_acc = self.evaluate(val_loader)
            
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Loss: {running_loss/len(train_loader):.4f} "
                  f"Train Acc: {train_acc:.2f}% "
                  f"Val Acc: {val_acc:.2f}%")
    
    # ---------- Evaluation ----------
    def evaluate(self, data_loader: DataLoader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in data_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
    
    # ---------- Save / Load ----------
    def save_model(self, path="googlenet_gesture.pth"):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path="googlenet_gesture.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()


# -----------------------------
# Class 2: Gesture Identifier
# -----------------------------
class GestureIdentifier:
    def __init__(self, architecture: GoogLeNetArchitecture, gesture_labels):
        self.architecture = architecture
        self.labels = gesture_labels
    
    def invoke(self, img: Image.Image):
        outputs = self.architecture.invoke(img)
        _, predicted = torch.max(outputs, 1)
        
        gesture = self.labels[predicted.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()
        
        return {
            'gesture': gesture,
            'confidence': confidence,
            'raw_outputs': outputs.tolist()
        }


# -----------------------------
# Example Usage (Training + Inference)
# -----------------------------
if __name__ == "__main__":
    # Assume dataset already built using ImageFolder
    from torchvision.datasets import ImageFolder
    from torch.utils.data import random_split
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageFolder("./gestures", transform=transform)
    num_classes = len(dataset.classes)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    # Initialize
    googlenet_arch = GoogLeNetArchitecture(num_classes=num_classes)
    gesture_identifier = GestureIdentifier(googlenet_arch, dataset.classes)
    
    # Train
    googlenet_arch.train_model(train_loader, val_loader, epochs=5, lr=0.001)
    googlenet_arch.save_model()
    
    # Inference test
    img = Image.open("./gestures/point/point_5.jpg")
    result = gesture_identifier.invoke(img)
    print(result)
