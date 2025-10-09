import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2

class GestureRecognizer:
    def __init__(self, model_path="googlenet_gesture.pth", labels=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = models.GoogLeNet_Weights.DEFAULT
        self.model = models.googlenet(weights=weights)
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
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(probs).item()
        return self.labels[pred], probs[pred].item()

if __name__ == "__main__":
    labels = ["pinch", "point"]  # your gesture labels
    recognizer = GestureRecognizer("googlenet_gesture.pth", labels)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        gesture, conf = recognizer.predict(img_pil)
        cv2.putText(frame, f"{gesture} ({conf:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
