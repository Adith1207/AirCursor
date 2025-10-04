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
    
    def get_finger_tips(self, mask, frame):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        frame_h, frame_w = frame.shape[:2]

        for c in contours:
            area = cv2.contourArea(c)
            if area < 2000 or area > 120000:  
                continue
            x, y, w, h = cv2.boundingRect(c)
            if y < frame_h // 6: 
                continue
            aspect_ratio = w / h
            if aspect_ratio > 1.5:  
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            if hull_area == 0 or area / hull_area < 0.65:  
                continue
            valid_contours.append(c)

        if len(valid_contours) == 0:
            return frame, []

        if self.multi_hand:
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        else:
            valid_contours = [max(valid_contours, key=cv2.contourArea)]

        results = []

        for contour in valid_contours:
            hull = cv2.convexHull(contour, returnPoints=False)
            if hull is None or len(hull) < 3:
                continue

            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

            fingertips = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = math.dist(start, end)
                b = math.dist(start, far)
                c = math.dist(end, far)
                if b * c == 0:
                    continue
                angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c))

                if angle <= math.pi / 2 and start[1] < cy and end[1] < cy and d > 10000:
                    fingertips.append(start)
                    fingertips.append(end)

            unique_fingers = []
            for pt in fingertips:
                if all(np.linalg.norm(np.array(pt) - np.array(p)) > 40 for p in unique_fingers):
                    unique_fingers.append(pt)

            unique_fingers.sort(key=lambda x: x[0])

            for pt in unique_fingers:
                cv2.circle(frame, pt, 8, (0, 0, 255), -1)

            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            results.append({
                "contour": contour,
                "fingertips": unique_fingers
            })

        return frame, results
    
    def release(self):
        self.capture.release()
        cv2.destroyAllWindows()

