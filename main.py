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
