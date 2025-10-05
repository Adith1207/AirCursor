import cv2
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# ------------------ CONFIG ------------------
MODEL_PATH = "googlenet_gesture.pth"   # Your trained model
GESTURE_LABELS = ["point", "pinch"]    # Make sure order matches training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 224
# -------------------------------------------

# 1️⃣ Define the GoogLeNet architecture class
class GoogLeNetArchitecture:
    def __init__(self, num_classes=2, pretrained=True, device=None):
        self.device = device or DEVICE
        self.model = models.googlenet(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    
    def invoke(self, img):
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        return out
    
    def load_model(self, path=MODEL_PATH):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

# 2️⃣ Gesture Identifier
class GestureIdentifier:
    def __init__(self, architecture, gesture_labels):
        self.arch = architecture
        self.labels = gesture_labels
    
    def invoke(self, img):
        outputs = self.arch.invoke(img)
        _, pred = torch.max(outputs, 1)
        gesture = self.labels[pred.item()]
        confidence = torch.softmax(outputs, dim=1)[0][pred.item()].item()
        return gesture, confidence

# 3️⃣ Initialize model
arch = GoogLeNetArchitecture(num_classes=len(GESTURE_LABELS))
arch.load_model(MODEL_PATH)
identifier = GestureIdentifier(arch, GESTURE_LABELS)

# 4️⃣ Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🎥 Webcam live gesture recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror for easier interaction
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    gesture, conf = identifier.invoke(pil_img)

    # Display prediction
    cv2.putText(frame, f"{gesture} ({conf*100:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Live Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
