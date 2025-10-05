import cv2
import os

# ========== CONFIG ==========
GESTURE_NAME = "point"   # 👈 Change this for each gesture ("point", "fist", "swipe", etc.)
NUM_SAMPLES = 200        # How many images you want to capture
SAVE_DIR = "gestures"    # Folder where images will be saved
# ============================

# Create gesture folder if not exists
gesture_path = os.path.join(SAVE_DIR, GESTURE_NAME)
os.makedirs(gesture_path, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print(f"🎥 Starting capture for gesture: {GESTURE_NAME}")
print("Press 's' to start saving images, 'q' to quit.")

count = 0
saving = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the camera
    cv2.putText(frame, f"Gesture: {GESTURE_NAME} | Saved: {count}/{NUM_SAMPLES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Capture Gestures", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):  # Start saving
        saving = True
        print("✅ Saving started...")

    if saving and count < NUM_SAMPLES:
        img_path = os.path.join(gesture_path, f"{GESTURE_NAME}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

        if count >= NUM_SAMPLES:
            print("🎉 Capture complete!")
            break

    if key == ord("q"):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
