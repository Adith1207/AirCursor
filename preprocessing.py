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
    
    def get_skinmask(self, frame):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        blur = cv2.GaussianBlur(ycrcb, (11, 11), 0)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(blur, lower_skin, upper_skin)
        return mask
    
    def morpho_mask(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.dilate(mask, self.kernel, iterations=1)
        return mask

