import csv
import time
import os
import numpy as np
import cv2
from pre_processing import WebcamStream


GESTURES = ["point", "pinch"]
SAMPLES_PER_GESTURE = 150
OUTPUT_CSV = "gesture_dataset.csv"


def extract_features(contour, fingertips):
    if contour is None or len(contour) < 3:
        return None

    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull) if len(hull) >= 3 else 0
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    rect_area = w * h
    extent = float(area) / rect_area if rect_area != 0 else 0
    solidity = float(area) / hull_area if hull_area != 0 else 0
    hull_ratio = hull_area / area if area != 0 else 0

    finger_count = len(fingertips) if fingertips is not None else 0

    avg_tip_dist = 0.0
    if finger_count > 1:
        dists = []
        for i in range(finger_count):
            for j in range(i + 1, finger_count):
                dists.append(np.linalg.norm(np.array(fingertips[i]) - np.array(fingertips[j])))
        avg_tip_dist = float(np.mean(dists)) if dists else 0.0

    return [
        finger_count,
        avg_tip_dist,
        aspect_ratio,
        extent,
        solidity,
        hull_ratio,
        area,
        perimeter
    ]