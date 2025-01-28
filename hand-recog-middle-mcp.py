import cv2
import mediapipe as mp
import math

#mediapipe init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

#calculate dist
def cal_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2 + (landmark1.z - landmark2.z) ** 2)


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
    
    hand_landmarks = {}
    if results.multi_hand_landmarks:
        for idx, hand_landmark in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            hand_landmarks[label] = hand_landmark

            # mp_draw.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)

    return results, image, hand_landmarks

#open cam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open cam")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Cannot catch frame")
        break

    results, frame, hand_landmarks = detect_hand_gesture(frame,hands)  

    if 'Left' in hand_landmarks and 'Right' in hand_landmarks:
        left_wrist = hand_landmarks['Left'].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        right_wrist = hand_landmarks['Right'].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        distance = cal_distance(left_wrist, right_wrist)
        
        left_wrist_pos = (int(left_wrist.x * frame.shape[1]), int(left_wrist.y * frame.shape[0]))
        right_wrist_pos = (int(right_wrist.x * frame.shape[1]), int(right_wrist.y * frame.shape[0]))

        cv2.circle(frame, left_wrist_pos, 5, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, right_wrist_pos, 5, (0, 255, 0), cv2.FILLED)

        text_pos = (left_wrist_pos[0], left_wrist_pos[1] - 20)
        cv2.putText(frame, f"Distance: {distance:.2f}", (text_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.line(frame, left_wrist_pos, right_wrist_pos, (255, 255, 255), 2)

    cv2.imshow("Hand gesture recog with mp and openCv",frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()