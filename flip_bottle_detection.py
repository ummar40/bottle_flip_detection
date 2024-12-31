import cv2
import numpy as np

# Initialize variables
player_scores = [0, 0]
current_player = 0
bottle_upright = False

# Bottle detection parameters
CAP_COLOR = (35, 100, 100) 
FLIP_THRESHOLD = 1.2  # Aspect ratio threshold for upright bottle
STABILITY_FRAMES = 10  # Frames required to confirm upright stability

# Initialize camera
cap = cv2.VideoCapture(0)  #use 0 for Built-in PC webcam

# Function to check if the bottle is upright based on contour
def is_bottle_upright(contour):
    rect = cv2.minAreaRect(contour)
    _, (width, height), _ = rect
    print(f"Width: {width}, Height: {height}")  # Debugging output for aspect ratio
    return height > width * FLIP_THRESHOLD

# Function to detect hands (for cheating detection)
def detect_hands(frame):
    """Placeholder for hand detection (optional for cheating)."""
    # For robust hand detection, integrate YOLOv5 or Haar Cascade.
    return False

# Frame processing variables
stable_frames = 0  # Stability frame counter

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Mask for green cap color detection
    # Adjust the HSV range to avoid detecting body parts
    mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))  # Green cap detection
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Detect contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bottle_detected = False  # Flag to ensure only bottle is detected

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small noise
            # Check if the contour is a bottle (upright) and not a hand or other object
            if is_bottle_upright(contour):
                stable_frames += 1
                print(f"Stable frames: {stable_frames}")  # Debugging output for stability frames
                if stable_frames >= STABILITY_FRAMES:
                    if not bottle_upright:
                        # Successful flip detected, update score
                        player_scores[current_player] += 1
                        print(f"Player {current_player + 1} scores! Current scores: {player_scores}")
                        current_player = 1 - current_player  # Switch player
                        bottle_upright = True
                    bottle_detected = True
            else:
                stable_frames = 0
                bottle_upright = False

    if not bottle_detected:
        stable_frames = 0
        bottle_upright = False

    # Detect cheating (e.g., hands near the bottle)
    if detect_hands(frame):
        print("Cheating detected! No points awarded.")
        stable_frames = 0
        bottle_upright = False

    # Display the current score and player's turn on the frame
    cv2.putText(frame, f"Player 1: {player_scores[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Player 2: {player_scores[1]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Player {current_player + 1}'s Turn", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the processed frame
    cv2.imshow("Bottle Flip Game", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
