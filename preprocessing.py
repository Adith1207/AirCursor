import cv2
import numpy as np
import math


class WebcamStream:
    def __init__(self, camindex=0, multi_hand=False):
        self.camindex = camindex
        self.capture = cv2.VideoCapture(self.camindex)

        if not self.capture.isOpened():
            raise Exception("Webcam not Opened")

        self.kernel = np.ones((5, 5), np.uint8)
        self.multi_hand = multi_hand  
    
    def read_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            raise Exception("Failed to capture the frame")
        return frame
