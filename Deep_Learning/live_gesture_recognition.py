import cv2
import torch
from torchvision import transforms
from PIL import Image
from train_model import GoogLeNetArchitecture, GestureIdentifier
from torchvision.datasets import ImageFolder

# Load gesture labels from your dataset
dataset = ImageFolder("./gestures", transform=transforms.ToTensor())
num_classes = len(dataset.classes)

# Initialize model
model_arch = GoogLeNetArchitecture(num_classes=num_classes, pretrained=False)
model_arch.load_model("googlenet_gesture.pth")

gesture_identifier = GestureIdentifier(model_arch, dataset.classes)

# Same transform used for validation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

cap = cv2.VideoCapture(0)
print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img)

    with torch.no_grad():
        outputs = model_arch.model(img.unsqueeze(0).to(model_arch.device))
        probs = torch.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(probs, 0)
        gesture = dataset.classes[predicted.item()]
        confidence = probs[predicted.item()].item()

    label = f"{gesture} ({confidence*100:.1f}%)"
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.imshow("Live Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
