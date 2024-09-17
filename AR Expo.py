import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize mediapipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Create a blank white canvas to draw on (similar to a paint app)
canvas = np.ones((580,740, 3), dtype=np.uint8) * 255  # White background

# Capture video input from webcam
cap = cv2.VideoCapture(0)

# Variables for drawing
drawing = False
prev_x, prev_y = None, None
color = (0, 0, 0)  # Default color: Black
brush_size = 10  # Fixed brush size
strokes = []  # List to store all strokes for undo functionality

# Color palette for user to select
color_palette = [(0, 0, 0), (0, 255, 0), (0, 0, 255)]  # Black, Green, Red

# Function to draw color palette on the frame
def draw_color_palette(frame):
    for i, col in enumerate(color_palette):
        cv2.rectangle(frame, (10 + i * 40, 10), (40 + i * 40, 40), col, -1)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hand landmarks
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the tip of the index finger (landmark 8) and thumb (landmark 4)
            index_finger_tip = hand_landmarks.landmark[8]
            middle_finger_tip = hand_landmarks.landmark[12]
            
            # Convert to pixel coordinates
            h, w, _ = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Color selection based on the number of raised fingers
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and hand_landmarks.landmark[12].y > hand_landmarks.landmark[11].y:  # 1 finger (black)
                color = (0, 0, 0)  # Black
            #elif hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y:  # 2 fingers (green)
                #color = (0, 255, 0)  # Green
            elif hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y:  # 3 fingers (red)
                color = (0, 0, 255)  # Red

            # Start drawing on the blank canvas when the index finger is up and middle finger is down
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y:  # Index finger tip above the joint
                if prev_x is None and prev_y is None:
                    prev_x, prev_y = cx, cy

                # Draw on the separate canvas (not on the video feed)
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), color, brush_size)
                prev_x, prev_y = cx, cy
                
                # Save the stroke for undo functionality
                strokes.append((prev_x, prev_y, cx, cy, color, brush_size))
            else:
                # Reset the starting point when the finger is not up
                prev_x, prev_y = None, None

            # Undo functionality (detect fist gesture)
            if (hand_landmarks.landmark[8].y > hand_landmarks.landmark[7].y and
                hand_landmarks.landmark[12].y > hand_landmarks.landmark[11].y and
                hand_landmarks.landmark[16].y > hand_landmarks.landmark[15].y):  # All fingers down (fist)
                if strokes:
                    strokes.pop()  # Remove last stroke
                    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Clear canvas
                    # Redraw all remaining strokes
                    for stroke in strokes:
                        cv2.line(canvas, (stroke[0], stroke[1]), (stroke[2], stroke[3]), stroke[4], stroke[5])

            # Optionally: Draw hand landmarks for debugging
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the video feed (for reference, not for drawing)
    cv2.imshow("Webcam", frame)
    
    # Draw color palette
    draw_color_palette(frame)
    
    # Display the drawing canvas in a separate window (as the painting app)
    cv2.imshow("Drawing Canvas", canvas)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
