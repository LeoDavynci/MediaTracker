import cv2
import mediapipe as mp
import pyautogui
import time
import math

class HandGestureController:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
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
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lm_list

    def recognize_gesture(self, lm_list, img):
        if not lm_list:
            return None
        # Gesture: Fist
        if self.is_fist(lm_list):
            return "Stop"
        # Gesture: Index finger up
        if lm_list[20][2] < lm_list[18][2]:
            return "play_pause"
        # Gesture: Volume Knob
        if self.volume_knob(lm_list, img):
            return "volume"

        return None

    def execute_action(self, gesture):
        if gesture == "play_pause":
            pyautogui.press('playpause')
            print("Play/Pause action executed")

    def is_fist(self, lm_list):
        # Check if all fingertips are below their respective MCP joints
        return all(lm_list[tip][2] > lm_list[mcp][2] for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)])

    def volume_knob(self, lm_list, img):
        # Calculate center point (pivot) using thumb, index, and middle fingertips
        thumbTip = lm_list[4]
        indexTip = lm_list[8]
        middleTip = lm_list[12]

        centerX = int((thumbTip[1] + indexTip[1] + middleTip[1]) / 3)
        centerY = int((thumbTip[2] + indexTip[2] + middleTip[2]) / 3)

        # Calculate angle of index finger relative to center point
        dx = indexTip[1] - centerX
        dy = indexTip[2] - centerY
        angle = math.degrees(math.atan2(dy, dx))

        # Normalize angle to 0-360 range
        angle = (angle + 360) % 360

        # Map angle to volume change (-100 to 100)
        volume_change = int(angle / 360 * 200) - 100

        # Draw center point
        cv2.circle(img, (centerX, centerY), 5, (255, 0, 0), cv2.FILLED)

        # Draw line from center to index fingertip
        cv2.line(img, (centerX, centerY), (indexTip[1], indexTip[2]), (0, 255, 0), 2)

        # Display angle and volume change
        cv2.putText(img, f"Angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Volume Change: {volume_change}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return volume_change

    def execute_action(self, gesture):
        current_time = time.time()
        if current_time - self.last_gesture_time < 2:  # 1 second cooldown
            return

        if gesture != "stop":

            if gesture == "play_pause":
                pyautogui.press('playpause')
                print("Play/Pause")

        else :
            print("No Actions")

        self.last_gesture_time = current_time

def main():
    cap = cv2.VideoCapture(0)
    controller = HandGestureController()
    p_time = 0

    while True:
        success, img = cap.read()
        img = controller.find_hands(img)
        lm_list = controller.find_position(img)

        if lm_list:
            gesture = controller.recognize_gesture(lm_list, img)
            if gesture:
                controller.execute_action(gesture)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()