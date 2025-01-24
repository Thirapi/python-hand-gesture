import cv2
import mediapipe as mp
import math

#mediapipe init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

#calculate dist
def cal_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2 + (landmark1.z - landmark2.z) ** 2)

#thumb index dist
def get_thumb_index_distance(result):
    if results.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            return cal_distance(thumb_tip, index_tip)
        return None

#calculate position
def cal_position(landmarkX, landmarkY):
    return int(landmarkX * frame.shape[1]), int(landmarkY * frame.shape[0])

#data position
def get_position(result):
    if results.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x, y = cal_position(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y)
            return x, y
        return None

#detect hand gest
def detect_hand_gesture(image,hand):
    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = hand.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return results, image

#open cam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open cam")
    exit()

while(cap.isOpened):
    ret, frame = cap.read()
    if not ret:
        print("Cannot catch frame")
        break

    results, frame = detect_hand_gesture(frame,hands)  

    distance = get_thumb_index_distance(results)
    if distance is not None:
        cv2.putText(frame, f"{distance:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        radius = int(distance * 250)
        x, y = get_position(results)
        color_intensity = int(distance * 255) 
        color = (255 - color_intensity, color_intensity, 0)
        cv2.circle(frame, (x, y), radius, color, -1)

    cv2.imshow("Hand gesture recog with mp and openCv",frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()