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
