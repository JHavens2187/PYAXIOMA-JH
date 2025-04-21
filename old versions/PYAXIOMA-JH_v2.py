import cv2
import numpy as np
import csv
import time
import math

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Configuration
current = input("Enter the current value (in A or other units): ")
filename = f"ball_angles_current_{current}.csv"
max_measurements = 200
measurement_delay = 0.05  # 50ms between measurements

# Tracking variables
measurement_count = 0
last_measurement_time = 0
recording = False
start_time = 0
tracking_object = None
selection_start = None
selection_rect = None
selecting = False

# Stationary circle variables
stationary_circle = None
setting_circle = False
circle_center = None
circle_radius = None

# Create CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)', 'Angle (degrees)', 'Current'])


def mouse_callback(event, x, y, flags, param):
    global selection_start, selection_rect, selecting, tracking_object
    global setting_circle, circle_center, circle_radius

    if setting_circle:
        if event == cv2.EVENT_LBUTTONDOWN:
            circle_center = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and circle_center is not None:
            dx = x - circle_center[0]
            dy = y - circle_center[1]
            circle_radius = int(math.sqrt(dx * dx + dy * dy))
        elif event == cv2.EVENT_LBUTTONUP:
            setting_circle = False
            print(f"Stationary circle set at {circle_center} with radius {circle_radius}")
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            selection_start = (x, y)
            selecting = True
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            x1, y1 = selection_start
            selection_rect = (min(x, x1), min(y, y1), abs(x - x1), abs(y - y1))
        elif event == cv2.EVENT_LBUTTONUP and selecting:
            selecting = False
            if selection_rect[2] > 5 and selection_rect[3] > 5:  # Minimum size
                # Create ROI mask
                mask = np.zeros_like(frame_gray)
                x, y, w, h = selection_rect
                mask[y:y + h, x:x + w] = 255

                # Find darkest area in selection
                _, thresholded = cv2.threshold(frame_gray[y:y + h, x:x + w], 0, 255,
                                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Find largest contour in selection
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"]) + x
                        cy = int(M["m01"] / M["m00"]) + y
                        tracking_object = {'center': (cx, cy), 'prev_center': (cx, cy)}
                        print(f"Tracking object set at ({cx}, {cy})")
            selection_rect = None


print("Instructions:")
print("1. Press 'c' to set the stationary circle (click center and drag for radius)")
print("2. Click and drag to SELECT the black object to track")
print("3. Press 's' to start/stop recording angles")
print("4. Press 'q' to quit")

cv2.namedWindow("Tracking View")
cv2.setMouseCallback("Tracking View", mouse_callback)

# For frame differencing
prev_gray = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    display_frame = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Hough Circle Transform for stationary circle detection
    if circle_center is not None and circle_radius is not None:
        circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=50, param2=30, minRadius=5, maxRadius=100)
        if circles is not None:
            circles = int(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(display_frame, (i[0], i[1]), i[2], (255, 0, 0), 2)
                cv2.circle(display_frame, (i[0], i[1]), 3, (255, 0, 0), -1)
        else:
            cv2.circle(display_frame, circle_center, circle_radius, (255, 0, 0), 2)
            cv2.circle(display_frame, circle_center, 3, (255, 0, 0), -1)

    # Show selection rectangle
    if selecting and selection_rect:
        x, y, w, h = selection_rect
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Track the selected object
    if tracking_object and circle_center is not None:
        # Use template matching to track the black object
        x, y = tracking_object['center']
        size = 30  # Tracking window size

        # Get ROI around previous position
        x1 = max(0, x - size)
        y1 = max(0, y - size)
        x2 = min(frame.shape[1], x + size)
        y2 = min(frame.shape[0], y + size)

        roi = frame_gray[y1:y2, x1:x2]

        # Threshold to find blackest area
        _, thresholded = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find largest contour in ROI
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1

                # Update tracking object
                tracking_object['prev_center'] = tracking_object['center']
                tracking_object['center'] = (cx, cy)

                # Draw tracking info
                cv2.circle(display_frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.putText(display_frame, "TRACKING", (cx + 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Calculate angle from stationary circle center to tracking object
                sc_x, sc_y = circle_center
                dx = cx - sc_x
                dy = cy - sc_y
                angle = math.degrees(math.atan2(dy, dx)) % 360

                # Draw angle line from stationary circle to tracking object
                cv2.line(display_frame, circle_center, (cx, cy), (0, 255, 0), 2)

                # Display angle
                cv2.putText(display_frame, f"{angle:.1f}°", (cx + 15, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Record data when enabled
                if recording and (time.time() - last_measurement_time) > measurement_delay:
                    elapsed = time.time() - start_time
                    with open(filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f"{elapsed:.3f}", f"{angle:.1f}", current])

                    measurement_count += 1
                    last_measurement_time = time.time()

                    if measurement_count >= max_measurements:
                        print(f"Completed {max_measurements} measurements")
                        recording = False

    # Display status information
    status = f"Recording: {measurement_count}/{max_measurements}" if recording else "Ready (press 's')"
    cv2.putText(display_frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Current: {current}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if tracking_object:
        cv2.putText(display_frame, "Tracking active", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "Drag to select object", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if circle_center is None:
        cv2.putText(display_frame, "Press 'c' to set stationary circle", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(display_frame, "Stationary circle set", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(display_frame, "'s' start/stop | 'q' quit", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show tracking view
    cv2.imshow("Tracking View", display_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if not recording and tracking_object and circle_center is not None:
            recording = True
            start_time = time.time()
            measurement_count = 0
            print("Recording started...")
        elif recording:
            recording = False
            print(f"Recording paused at {measurement_count} measurements")
    elif key == ord('c'):
        setting_circle = True
        circle_center = None
        circle_radius = None
        print("Click and drag to set stationary circle")

cap.release()
cv2.destroyAllWindows()
print(f"Data collection complete. Saved to {filename}")
