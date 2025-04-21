import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import Toplevel, Button, Label
import time
from collections import deque


# Create a figure and an axes for the graph
fig, ax = plt.subplots()
line, = ax.plot([], [])  # Create an empty line object
x_data, y_data = [], []  # Initialize data for plotting

# Initialize video capture
#cap = cv2.VideoCapture('output_video.mp4')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Initialize variables
ball_center = None
radius = None
prev_gray = None  # For optical flow tracking
calibrating = True  # Flag for calibration phase
motion_threshold = None  # Default motion threshold
best_circle = False  # Flag for cue ball detection
angle_values = []  # List to store angle values for plotting

# Create main window
root = tk.Tk()
root.title("Tracking System")

# Create a frame for the video feeds
frame = tk.Frame(root)
frame.pack()

# Create labels for data display
elapsed_time_label = Label(root, text="Elapsed Time: 0s")
elapsed_time_label.pack()
timestamp_label = Label(root, text="Timestamp (UTC): ")
timestamp_label.pack()
fps_label = Label(root, text="FPS: 0")
fps_label.pack()
angular_velocity_label = Label(root, text="Angular Velocity: 0")
angular_velocity_label.pack()
confidence_score_label = Label(root, text="Confidence Score: 0")
confidence_score_label.pack()


# Function to open settings window
def open_settings():
    settings_window = Toplevel(root)
    settings_window.title("Settings")
    settings_label = Label(settings_window, text="Settings Panel")
    settings_label.pack()
    # Example setting: a simple close button for the settings panel
    close_button = Button(settings_window, text="Close", command=settings_window.destroy)
    close_button.pack(pady=10)


settings_button = Button(root, text="Settings", command=open_settings)
settings_button.pack()

cv2.namedWindow("Live Feed", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Gray Feed", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Optical Flow Motion", cv2.WINDOW_AUTOSIZE)


# Update function for the graph
def update(frame):
    if angle_values:
        # Skip NaN values
        angle = angle_values[-1]
        if not np.isnan(angle):
            x_data.append(frame)
            y_data.append(angle)
            line.set_data(x_data, y_data)
            ax.relim()
            ax.autoscale_view()
    return line,


# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 1000), blit=True)


def find_black_piece_and_angle(frame, bounding_box, circle_center):
    x, y, w, h = bounding_box
    roi_black = frame[y:y + h, x:x + w]

    # Convert to HSV and create mask for black color
    hsv_black = cv2.cvtColor(roi_black, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])
    mask_black = cv2.inRange(hsv_black, lower_black, upper_black)

    # Find contours in the black mask
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_black:
        largest_contour = max(contours_black, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        print(M)
        if M["m00"] != 0:
            cx_black = int(M["m10"] / M["m00"]) + x
            cy_black = int(M["m01"] / M["m00"]) + y
            print("cy and cx are: ", cy_black, cx_black)

            # Calculate angle
            dx = cx_black - circle_center[0]
            dy = cy_black - circle_center[1]
            angle = math.atan2(dy, dx) * (180 / np.pi)

            # Draw indicators
            cv2.circle(frame, (cx_black, cy_black), 5, (0, 255, 0), -1)  # Black piece centroid
            cv2.line(frame, circle_center, (cx_black, cy_black), (255, 0, 0), 2)  # Line to black piece

            return angle, cx_black, cy_black


# Detect the cue ball once
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2

    # Define initial ROI
    roi_size = min(frame_width, frame_height) // 3
    roi_x1, roi_y1 = center_x - roi_size // 2, center_y - roi_size // 2
    roi_x2, roi_y2 = center_x + roi_size // 2, center_y + roi_size // 2
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # Convert to HSV and create mask for cue ball
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Convert masked ROI to grayscale
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
    gray_masked = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)

    # Detect cue ball using Hough Circle Transform
    circles = cv2.HoughCircles(gray_masked, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=10, maxRadius=100)

    if circles is not None:
        circles = int(np.around(circles))
        min_dist = float('inf')
        best_circle = None

        for i in circles[0, :]:
            x, y, r = i[0] + roi_x1, i[1] + roi_y1, i[2]
            dist_to_center = abs(y - center_y)

            if dist_to_center < min_dist:
                min_dist = dist_to_center
                best_circle = (x, y, r)

        if best_circle:
            ball_center = (best_circle[0], best_circle[1])
            radius = best_circle[2]
            best_circle = True

            # Define new ROI centered on the cue ball
            cx, cy = ball_center
            roi_x1, roi_y1 = max(cx - roi_size // 2, 0), max(cy - roi_size // 2, 0)
            roi_x2, roi_y2 = min(cx + roi_size // 2, frame_width), min(cy + roi_size // 2, frame_height)
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    if best_circle:
        break

# Main loop for processing
frame_count = 0
bounding_box = None
start_time = time.time()
while True:
    angle = None
    ret, frame = cap.read()
    if not ret:
        break

    if ball_center is not None:
        cx, cy = ball_center

        # Define new ROI centered on the cue ball
        roi_x1, roi_y1 = max(cx - roi_size // 2, 0), max(cy - roi_size // 2, 0)
        roi_x2, roi_y2 = min(cx + roi_size // 2, frame_width), min(cy + roi_size // 2, frame_height)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Convert ROI to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ==== Optical Flow for Tracking Motion ====
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Highlight moving areas based on calibrated threshold
            if motion_threshold is None:
                motion_threshold = 1.75
            moving_mask = np.uint8(mag > motion_threshold) * 255

            # Find contours in motion mask
            motion_contours, _ = cv2.findContours(moving_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in motion_contours:
                x_m, y_m, w_m, h_m = cv2.boundingRect(cnt)
                if w_m > 10 and h_m > 10:  # Filter small noise
                    x_m += roi_x1
                    y_m += roi_y1
                    bounding_box = (x_m, y_m, w_m, h_m)
                    cv2.rectangle(frame, (x_m, y_m), (x_m + w_m, y_m + h_m), (0, 255, 255), 2)
                    cv2.rectangle(moving_mask, (x_m, y_m), (x_m + w_m, y_m + h_m), (0, 255, 255), 2)

                    # Call function to detect black piece and calculate angle
                    angle, cx_black, cy_black = find_black_piece_and_angle(frame, bounding_box, ball_center)
                    # add angle data to live feed
                    if angle:
                        angle_values.append(angle)  # Store angle for plotting
                        cv2.line(frame, ball_center, (cx + radius, cy), (255, 0, 0), 2)  # Line to black piece
                        cv2.putText(frame, f"Angle: {angle:.2f} degrees", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        cv2.putText(frame, "TRACKING ACTIVE", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the motion mask
            cv2.circle(moving_mask, (cx, cy), 2, (0, 0, 255), 3)
            cv2.imshow("Optical Flow Motion", moving_mask)

        # Store current bounding box and current frame for next iteration
        #previous_box = bounding_box
        prev_gray = gray.copy()

        # Draw cue ball circle on all feeds
        cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(gray, (cx, cy), 2, (0, 0, 255), 2)

        # Display frames
        if angle is None:
            cv2.putText(frame, "Angle: NaN", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "TRACKING INACTIVE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Live Feed", frame)
        cv2.imshow("Gray Feed", gray)


        # Function to calculate angular velocity
        def calculate_angular_velocity(angle_values, fps):
            if len(angle_values) < 2:
                return 0
            return (angle_values[-1] - angle_values[-2]) / (1 / fps)


        # Function to calculate smoothness of angular velocity
        def calculate_smoothness(angular_velocities):
            if len(angular_velocities) < 2:
                return 1  # Default high smoothness for initial values
            diffs = np.diff(angular_velocities)
            smoothness = np.std(diffs)
            return smoothness


        angular_velocity_buffer = deque(maxlen=10)  # Adjust window size as needed


        def smooth_angular_velocity(new_value):
            angular_velocity_buffer.append(new_value)
            return np.mean(angular_velocity_buffer)  # Simple moving average


        # Function to calculate confidence score
        def calculate_iou(box1, box2):
            """
            Calculates the Intersection over Union (IoU) of two bounding boxes.

            Args:
            - box1: (x1, y1, x2, y2) for the first bounding box
            - box2: (x1, y1, x2, y2) for the second bounding box

            Returns:
            - IoU value
            """
            # Coordinates of the intersection box
            x1_inter = max(box1[0], box2[0])
            y1_inter = max(box1[1], box2[1])
            x2_inter = min(box1[2], box2[2])
            y2_inter = min(box1[3], box2[3])

            # Area of intersection
            intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

            # Area of both bounding boxes
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

            # Area of union
            union_area = box1_area + box2_area - intersection_area

            # IoU calculation
            iou = intersection_area / union_area if union_area > 0 else 0
            return iou


        def temporal_consistency(prev_center, prev_radius, current_center, current_radius, threshold=10):
            if prev_center is None or prev_radius is None:
                return True  # No previous data to compare with

            # Calculate the change in center and radius
            center_change = np.linalg.norm(np.array(current_center) - np.array(prev_center))
            radius_change = abs(current_radius - prev_radius)

            if center_change > threshold or radius_change > threshold:
                return False  # The changes are too abrupt
            return True


        def weighted_average_confidence(iou, consistency, weight_iou=0.7, weight_consistency=0.3):
            return iou * weight_iou + consistency * weight_consistency


        angular_velocity = calculate_angular_velocity(angle_values, int(cap.get(cv2.CAP_PROP_FPS)))

        # Calculate IoU
        if bounding_box:
                _, cx_black, cy_black = find_black_piece_and_angle(frame, bounding_box, ball_center)
                box1 = (cx - radius, cy - radius, cx + radius, cy + radius)  # Example bounding box for cue ball
                box2 = (cx_black - 10, cy_black - 10, cx_black + 10, cy_black + 10)  # Example bounding box for black piece

                iou = calculate_iou(box1, box2)

                # Calculate temporal consistency with the previous frame's data
                consistency = temporal_consistency(prev_center, prev_radius, (cx, cy), radius)

                # Compute weighted average confidence score
                confidence = weighted_average_confidence(iou, consistency)

                confidence_score_label.config(text=f"Confidence Score: {confidence}")

        # Update data labels
        elapsed_time = time.time() - start_time
        elapsed_time_label.config(text=f"Elapsed Time: {elapsed_time:.1f}s")
        timestamp_label.config(text=f"Timestamp (UTC): {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
        fps_label.config(text=f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
        smoothed_angular_velocity = smooth_angular_velocity(angular_velocity)
        angular_velocity_label.config(text=f"Angular Velocity: {smoothed_angular_velocity:.2f} rad/s")
    root.update_idletasks()
    root.update()

    # Press 'q' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
root.mainloop()
