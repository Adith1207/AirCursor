import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class GestureRecognizer:
    def __init__(self, model_path="googlenet_gesture.pth", labels=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = models.GoogLeNet_Weights.DEFAULT
        # Keep aux_logits=True to match pretrained weights
        self.model = models.googlenet(weights=weights)  # aux_logits=True
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(labels))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.labels = labels

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, img):
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            # Take main output if aux_logits=True
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(probs).item()
        return self.labels[pred], probs[pred].item()


def evaluate_model(model_path, test_dir, labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    dataset = ImageFolder(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    recognizer = GestureRecognizer(model_path=model_path, labels=labels)

    y_true = []
    y_pred = []

    for images, targets in loader:
        images = images.to(device)
        with torch.no_grad():
            output = recognizer.model(images)
            if isinstance(output, tuple):
                output = output[0]  # main output only
            preds = torch.argmax(output, dim=1)
        y_true.append(targets.item())
        y_pred.append(preds.item())

    # Metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    labels = ["pinch", "point"]  # change according to your classes
    test_dir = "test_dataset"     # folder containing test/<class_name>/*.jpg
    model_path = "googlenet_gesture.pth"

    evaluate_model(model_path, test_dir, labels)
