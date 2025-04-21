import cv2
import numpy as np
import time
import math

# Initialize video capture from the default camera (0)
cap = cv2.VideoCapture('raw_video/0.5Amps.avi')

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for cue ball detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Detect cue ball using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=20, maxRadius=100)

    ball_center = None
    if circles is not None:
        circles = int(np.around(circles))
        for i in circles[0, :]:
            ball_center = (i[0], i[1])
            radius = i[2]
            cv2.circle(frame, ball_center, radius, (0, 255, 0), 2)  # Green circle around cue ball
            cv2.circle(frame, ball_center, 2, (0, 0, 255), 3)  # Red dot at center

    # Edge detection for black piece detection
    edges = cv2.Canny(gray, 50, 150)

    # Apply morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Find contours in edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_piece_center = None
    if ball_center is not None:
        cx, cy = ball_center
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                # Ensure detected object is within the cue ball's region
                if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                    black_piece_center = (x, y)
                    cv2.circle(frame, black_piece_center, 5, (255, 0, 0), -1)  # Blue dot on black piece
                    cv2.line(frame, ball_center, black_piece_center, (255, 0, 0), 2)  # Line from center through black piece

    # Calculate and display angle
    if ball_center is not None and black_piece_center is not None:
        x1, y1 = ball_center
        x2, y2 = black_piece_center
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        cv2.putText(frame, f"Angle: {angle:.2f} degrees", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display frames
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Edges", edges)
    cv2.imshow("Gray", gray)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
