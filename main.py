import cv2
import mediapipe as mp
import time
import json
import asyncio
import websockets
import threading

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

current_gesture = None
is_paused = False


async def detect_fingers_and_send(websocket, path):
    global current_gesture, is_paused
    while True:
        if current_gesture and not is_paused:
            await websocket.send(json.dumps({"gesture": current_gesture}))
            is_paused = True
            await asyncio.sleep(3)
            is_paused = False
        await asyncio.sleep(0.1)
        

def detect_fingers_raised(hand_landmarks):
    global is_paused
    if is_paused:
        return None
    
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

    fingers_raised = {
        'index': index_tip.y < index_dip.y,
        'middle': middle_tip.y < middle_dip.y,
        'ring': ring_tip.y < ring_dip.y,
        'pinky': pinky_tip.y < pinky_dip.y,
        'thumb': thumb_tip.x > wrist.x
    }

    if fingers_raised['index'] and not any([fingers_raised['middle'], fingers_raised['ring'], fingers_raised['pinky']]):
        return 'Selected answer: One'
    elif fingers_raised['index'] and fingers_raised['middle'] and not any([fingers_raised['ring'], fingers_raised['pinky']]):
        return 'Selected answer: Two'
    elif fingers_raised['index'] and fingers_raised['middle'] and fingers_raised['ring'] and not fingers_raised['pinky']:
        return 'Selected answer: Three'
    elif fingers_raised['index'] and fingers_raised['middle'] and fingers_raised['ring'] and fingers_raised['pinky']:
        return 'Selected answer: Four'
    elif fingers_raised['thumb'] and not any([fingers_raised['index'], fingers_raised['middle'], fingers_raised['ring'], fingers_raised['pinky']]):
        return 'Submit'
    elif fingers_raised['pinky'] and not any([fingers_raised['index'], fingers_raised['middle'], fingers_raised['ring'], fingers_raised['thumb']]):
        return 'Return'
    elif fingers_raised['index'] and fingers_raised['pinky'] and not any([fingers_raised['ring'], fingers_raised['middle']]):
        return 'Previous Tab'
    elif fingers_raised['ring'] and fingers_raised['pinky'] and not any([fingers_raised['index'], fingers_raised['middle']]):
        return 'Next Tab'
    
    return None

def webcam_thread():
    global current_gesture
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_gesture = detect_fingers_raised(hand_landmarks)
        
        else:
            current_gesture = None

        if current_gesture:
            cv2.putText(frame, current_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
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
        if cv2.waitKey(1) & 0xFF == ord('x'):  # Press 'x' to exit
            break
    cap.release()
    cv2.destroyAllWindows


async def main():
    webcam = threading.Thread(target=webcam_thread)
    webcam.start()

    async with websockets.serve(detect_fingers_and_send, '192.168.1.3', 8765):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped")

