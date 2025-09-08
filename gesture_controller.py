import math
import pyautogui

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

class GestureController:
    def __init__(self, sensitivity=2.5, dead_zone=5):
        self.prev_finger_pos = None
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone
        self.dragging = False