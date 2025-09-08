import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, track_conf=0.7):
        self.max_hands = max_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=track_conf
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.locked_hand_landmarks = None
    
    def find_hands(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks:
            self.locked_hand_landmarks = self.results.multi_hand_landmarks[0]

            if draw:
                self.mp_draw.draw_landmarks(
                    frame, self.locked_hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        else:
            self.locked_hand_landmarks = None

        return frame