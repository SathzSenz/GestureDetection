import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def detect_fingers_raised(hand_landmarks):
    # Get the landmarks for each finger
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]

    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]

    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]

    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]

    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Define the raised fingers
    fingers_raised = {
        'index': index_tip.y < index_dip.y,
        'middle': middle_tip.y < middle_dip.y,
        'ring': ring_tip.y < ring_dip.y,
        'pinky': pinky_tip.y < pinky_dip.y,
        'thumb': thumb_tip.x > wrist.x  # Thumb raised if to the right of the wrist
    }

    # Determine the gesture based on raised fingers
    # Answer one gesture (index only)
    if fingers_raised['index'] and not any([fingers_raised['middle'], fingers_raised['ring'], fingers_raised['pinky']]):
        return 'Selected answer: One'
    # Answer two gesture (index and middle)
    elif fingers_raised['index'] and fingers_raised['middle'] and not any([fingers_raised['ring'], fingers_raised['pinky']]):
        return 'Selected answer: Two'
    # Answer three gesture (index, middle, and ring)
    elif fingers_raised['index'] and fingers_raised['middle'] and fingers_raised['ring'] and not fingers_raised['pinky']:
        return 'Selected answer: Three'
    # Answer four gesture (all fingers except thumb)
    elif fingers_raised['index'] and fingers_raised['middle'] and fingers_raised['ring'] and fingers_raised['pinky']:
        return 'Selected answer: Four'
    # Submit gesture (thumb only)
    elif fingers_raised['thumb'] and not any([fingers_raised['index'], fingers_raised['middle'], fingers_raised['ring'], fingers_raised['pinky']]):
        return 'Submit'
    # Return gesture (pinky only)
    elif fingers_raised['pinky'] and not any([fingers_raised['index'], fingers_raised['middle'], fingers_raised['ring'], fingers_raised['thumb']]):
        return 'Return'
    # Next gesture (thumb and index only)
    elif fingers_raised['index'] and fingers_raised['pinky'] and not any([fingers_raised['ring'], fingers_raised['middle']]):
        return 'Previous Tab'  # Gesture for previous tab
    
    elif fingers_raised['ring'] and fingers_raised['pinky'] and not any([fingers_raised['index'], fingers_raised['middle']]):
        return 'Next Tab'  # Gesture for next tab
    
    return None


fps = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_fingers_raised(hand_landmarks)

    if gesture:
        cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    frame_count += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = end_time

    frame_width = frame.shape[1]
    fps_text_position = (frame_width - 150, 30)

    cv2.putText(frame, f"FPS: {fps:.2f}", fps_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()