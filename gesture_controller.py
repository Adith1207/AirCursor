import math
import pyautogui

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

class GestureController:
    def __init__(self, sensitivity=2.5, dead_zone=5):
        self.prev_finger_pos = None
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone
        self.dragging = False
    
    @staticmethod
    def distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.hypot(x2 - x1, y2 - y1)
    
    def map_to_screen(self, raw_x, raw_y, cam_width, cam_height):
        screen_x = (raw_x / cam_width) * SCREEN_WIDTH
        screen_y = (raw_y / cam_height) * SCREEN_HEIGHT
        return int(screen_x), int(screen_y)
    
    def smooth_position(self, current_pos):
        if self.prev_finger_pos is None:
            self.prev_finger_pos = current_pos
            return current_pos