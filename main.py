import cv2
import numpy as np
import mediapipe as mp
from pynput.keyboard import Controller, Key
import vlc

# Initialize VLC player
vlc_instance = vlc.Instance()
player = vlc_instance.media_player_new()

# Open connection to webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()       
mp_drawing = mp.solutions.drawing_utils

# Initialize the keyboard controller
keyboard = Controller()

volume = 50  # Initial volume level
player.audio_set_volume(volume)

# Variables to track previous palm positions
prev_palm_y = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get hand landmark coordinates
            landmarks = hand_landmarks.landmark
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = landmarks[mp_hands.HandLandmark.WRIST]

            # Convert to pixel coordinates
            thumb_tip_x = int(thumb_tip.x * frame.shape[1])
            thumb_tip_y = int(thumb_tip.y * frame.shape[0])
            index_tip_x = int(index_tip.x * frame.shape[1])
            index_tip_y = int(index_tip.y * frame.shape[0])
            wrist_x = int(wrist.x * frame.shape[1])
            wrist_y = int(wrist.y * frame.shape[0])

            # Check if thumb is extended for play
            if thumb_tip_y < index_tip_y and abs(thumb_tip_x - index_tip_x) < 50:
                print("Thumb detected - Playing")
                keyboard.press(Key.space)  # Simulate 'play' (space bar)
                keyboard.release(Key.space)
                
            # Check if index finger is extended for pause
            if index_tip_y < thumb_tip_y and abs(index_tip_x - thumb_tip_x) < 50:
                print("Index finger detected - Pausing")
                keyboard.press(Key.space)  # Simulate 'pause' (space bar)
                keyboard.release(Key.space)
            
            # Check for palm gesture for volume control
            if prev_palm_y is not None:
                if wrist_y < prev_palm_y - 30:
                    print("Palm dragged up - Increasing volume")
                    volume = min(volume + 10, 100)
                    player.audio_set_volume(volume)
                elif wrist_y > prev_palm_y + 30:
                    print("Palm dragged down - Decreasing volume")
                    volume = max(volume - 10, 0)
                    player.audio_set_volume(volume)

            prev_palm_y = wrist_y

    cv2.imshow('Frame', frame)  # Display the resulting frame

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera capture and close all windows
cap.release()
cv2.destroyAllWindows()
