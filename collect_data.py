import csv
import time
import os
import numpy as np
import cv2
from pre_processing import WebcamStream


GESTURES = ["point", "pinch"]
SAMPLES_PER_GESTURE = 150
OUTPUT_CSV = "gesture_dataset.csv"
