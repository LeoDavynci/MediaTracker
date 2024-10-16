import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class HandGestureController:
    def __init__(self, mode=False, max_hands=1, detection_con=0.8, track_con=0.8):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_gesture_time = time.time()

        # Volume control initialization
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.vol_range = self.volume.GetVolumeRange()
        self.min_vol = self.vol_range[0]
        self.max_vol = self.vol_range[1]
        self.vol = 0
        self.vol_bar = 400
        self.vol_perc = 0

        # Swipe detection variables
        self.prev_positions = []
        self.swipe_detection_window = 4
        self.swipe_threshold = 50

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            try:
                my_hand = self.results.multi_hand_landmarks[hand_no]
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            except IndexError:
                pass
        return lm_list

    def recognize_gesture(self, lm_list, img):
        if not lm_list:
            return None

        # Gesture: Thumb and Pinky Out for Play/Pause
        if self.is_thumb_pinky_out(lm_list):
            return "play_pause"

        # Gesture: Swipe gestures
        swipe_gesture = self.detect_swipe(lm_list, img)
        if swipe_gesture:
            return swipe_gesture

        return None

    def execute_action(self, gesture):
        current_time = time.time()
        if current_time - self.last_gesture_time < 1:
            return

        if gesture == "play_pause":
            pyautogui.press('playpause')
            print("Play/Pause")
        elif gesture == "swipe_right":
            pyautogui.press('nexttrack')
            print("Next Track")
        elif gesture == "swipe_left":
            pyautogui.press('prevtrack')
            pyautogui.press('prevtrack')
            print("Previous Track")

        self.last_gesture_time = current_time

    def is_thumb_pinky_out(self, lm_list):
        # Get the finger states
        fingers = self.get_finger_states(lm_list)

        # Check if thumb and pinky are up, and other fingers are down
        if fingers == [1, 0, 0, 0, 1]:
            return True
        return False

    def get_finger_states(self, lm_list):
        fingers = []
        # Thumb: Use x-coordinate for thumb (left-right movement)
        if lm_list[4][1] > lm_list[3][1]:  # Right hand thumb to the right
            fingers.append(1)
        else:
            fingers.append(0)
        # Index Finger
        if lm_list[8][2] < lm_list[6][2]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Middle Finger
        if lm_list[12][2] < lm_list[10][2]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Ring Finger
        if lm_list[16][2] < lm_list[14][2]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Pinky Finger
        if lm_list[20][2] < lm_list[18][2]:
            fingers.append(1)
        else:
            fingers.append(0)
        return fingers

    def detect_swipe(self, lm_list, img):
        # Check if the hand is open
        if not self.is_hand_open(lm_list):
            # Reset positions if hand is not open
            self.prev_positions = []
            return None

        # Calculate the center point of the hand
        x_values = [lm[1] for lm in lm_list]
        y_values = [lm[2] for lm in lm_list]
        center_x = int(sum(x_values) / len(x_values))
        center_y = int(sum(y_values) / len(y_values))

        # Draw a circle at the center point
        cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), cv2.FILLED)

        # Add the current center position to the positions list
        self.prev_positions.append((center_x, center_y))

        # Keep only the last N positions
        if len(self.prev_positions) > self.swipe_detection_window:
            self.prev_positions.pop(0)

        # If we have enough positions, check for swipe
        if len(self.prev_positions) == self.swipe_detection_window:
            # Calculate movement
            delta_x = self.prev_positions[-1][0] - self.prev_positions[0][0]
            delta_y = self.prev_positions[-1][1] - self.prev_positions[0][1]

            # Check if movement is mostly horizontal
            if abs(delta_x) > self.swipe_threshold and abs(delta_y) < self.swipe_threshold / 2:
                # Reset positions after detecting swipe
                self.prev_positions = []
                if delta_x > 0:
                    print("Swipe Left Detected")
                    return "swipe_left"
                else:
                    print("Swipe Right Detected")
                    return "swipe_right"
        return None

    def is_hand_open(self, lm_list):
        # Check if all fingers are up
        fingers = self.get_finger_states(lm_list)
        return fingers == [1, 1, 1, 1, 1]

    def adjust_volume_with_pinch(self, lm_list, img):
        if len(lm_list) >= 21:
            # Check the state of all fingers
            fingers = self.get_finger_states(lm_list)

            # Check if thumb and index finger are up, and other fingers are down
            if fingers == [1, 1, 0, 0, 0]:
                # Proceed with volume adjustment
                x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb tip
                x2, y2 = lm_list[8][1], lm_list[8][2]  # Index finger tip

                # Draw circles on thumb tip and index finger tip
                cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)

                # Draw line between thumb and index finger tips
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                # Calculate distance between thumb and index finger tips
                length = math.hypot(x2 - x1, y2 - y1)

                # Adjusted mapping: reduce the required length range
                vol = np.interp(length, [20, 150], [self.min_vol, self.max_vol])
                self.vol_bar = np.interp(length, [20, 150], [400, 150])
                self.vol_perc = np.interp(length, [20, 150], [0, 100])

                # Limit volume to the min and max values
                vol = max(self.min_vol, min(self.max_vol, vol))

                # Set the volume
                self.volume.SetMasterVolumeLevel(vol, None)

                # Draw volume bar
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
                cv2.rectangle(img, (50, int(self.vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(self.vol_perc)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 255, 0), 3)
                return True
        return False

def main():
    cap = cv2.VideoCapture(0)
    controller = HandGestureController()
    p_time = 0

    while True:
        success, img = cap.read()
        img = controller.find_hands(img)
        lm_list = controller.find_position(img, draw=False)

        if lm_list:
            # Adjust volume with pinch gesture
            if controller.adjust_volume_with_pinch(lm_list, img):
                pass  # Volume adjusted
            else:
                # Recognize gestures
                gesture = controller.recognize_gesture(lm_list, img)
                if gesture:
                    controller.execute_action(gesture)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
