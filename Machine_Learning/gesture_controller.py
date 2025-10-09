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
        # → Converts camera coordinates to screen resolution (linear interpolation).

        screen_x = (raw_x / cam_width) * SCREEN_WIDTH
        screen_y = (raw_y / cam_height) * SCREEN_HEIGHT
        return int(screen_x), int(screen_y)
    
    def smooth_position(self, current_pos): 
        # → Applies smoothing to prevent jitter.

        if self.prev_finger_pos is None:
            self.prev_finger_pos = current_pos
            return current_pos
        
        dx = (current_pos[0] - self.prev_finger_pos[0]) * self.sensitivity
        dy = (current_pos[1] - self.prev_finger_pos[1]) * self.sensitivity

        if abs(dx) < self.dead_zone and abs(dy) < self.dead_zone:
            return self.prev_finger_pos

        new_x = max(0, min(SCREEN_WIDTH, self.prev_finger_pos[0] + int(dx)))
        new_y = max(0, min(SCREEN_HEIGHT, self.prev_finger_pos[1] + int(dy)))

        self.prev_finger_pos = (new_x, new_y)
        return (new_x, new_y)
    
    def move_cursor(self, x, y):
        pyautogui.moveTo(x, y)
    
    def handle_drag_drop(self, classifier_output):
        '''
            Kept the same logic, but simplified to use pyautogui.mouseDown() and mouseUp() instead of low-level uinput calls.
        '''
        
        if classifier_output == "pinch" and not self.dragging:
            pyautogui.mouseDown()
            self.dragging = True
        elif classifier_output != "pinch" and self.dragging:
            pyautogui.mouseUp()
            self.dragging = False

