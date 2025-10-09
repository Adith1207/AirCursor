import cv2
import time
from hand_tracker import HandTracker
from gesture_controller import GestureController
import numpy as np

class Classifier:
    def get_output(self):
        """
        Placeholder: Replace this with your actual model inference logic.
        Should return "pinch" or "open"
        """
        return np.random.choice(["pinch", "open"])  # Example random output

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    controller = GestureController(sensitivity=2.5, dead_zone=5)
    classifier = Classifier()

    cam_width = 640
    cam_height = 480

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)  # Mirror camera
            frame = tracker.find_hands(frame, draw=True)

            fingertip = tracker.get_fingertip_coordinates(frame)

            if fingertip:
                # Map to screen resolution
                screen_x, screen_y = controller.map_to_screen(
                    fingertip[0], fingertip[1], cam_width, cam_height
                )

                # Smooth position
                smooth_x, smooth_y = controller.smooth_position((screen_x, screen_y))

                # Move cursor
                controller.move_cursor(smooth_x, smooth_y)

                # Get classifier output
                pinch_state = classifier.get_output()

                # Handle drag-drop based on pinch
                controller.handle_drag_drop(pinch_state)

            cv2.imshow("System Integration", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

            time.sleep(0.01)  # Small delay

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
