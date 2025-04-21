import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import pandas as pd
from datetime import datetime, timezone
import tkinter as tk
from tkinter import Label, Button, Toplevel

# Create a figure and an axes
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')  # Red line
x_data, y_data = [], []
frame_count = 0  # Track frames manually
Time = []
UTC_time = []
confidence_values = []  # New list for confidence values

# Initialize video capture
#cap = cv2.VideoCapture('raw_video/0.5Amps.avi')
#cap = cv2.VideoCapture('output_video.avi')
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
#direction = int(input("90 or straight? "))
direction = 0
amps = float(input("enter the Amp value: "))

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

def update(frame):
    global frame_count

    if angle_values:
        angle = angle_values[-1]  # Get the most recent angle
        if not np.isnan(angle):
            x_data.append(frame_count)
            y_data.append(angle)
            line.set_data(x_data, y_data)

            # Auto-adjust axes
            ax.relim()
            ax.autoscale_view()

            frame_count += 1  # Increment frame count

    return line,


def OpenCV_dbprint(msg, loc, color=(0, 0, 255)):
    """Prints a message on the OpenCV window.
    Args:
        msg (str): The message to print.
        loc (tuple): The location (x, y) to print the message.
        color (tuple): The color of the text in BGR format.
        """
    cv2.putText(frame, msg, loc, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


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
        if M["m00"] != 0:
            cx_black = int(M["m10"] / M["m00"]) + x
            cy_black = int(M["m01"] / M["m00"]) + y

            # Calculate angle
            Bx = int(cx_black - circle_center[0])
            By = int(cy_black - circle_center[1])
            if Bx > 32767:  # Adjust for 16-bit signed integer overflow when python hates me
                Bx = Bx - 2 ** 16

            if By > 32767:  # Adjust for 16-bit signed integer overflow when python hates me
                By = By - 2 ** 16

            By *= -1  # Invert Y-axis (OpenCV puts (0,0) at the top-left corner, with y increasing downward)

            #print(f"Bx: {Bx}, By: {By}")

            # Compute the dot product with the reference vector (r,0)
            dot_product = Bx  # Since ref vector is (r,0), this simplifies to just Bx
            if Bx < 0:
                dot_product *= -1  # Reflect across vertical axis for quadrant II
            mag_vecB = np.sqrt(Bx ** 2 + By ** 2)  # Magnitude of vector B

            # Calculate angle using arccos
            cos_theta = dot_product / mag_vecB
            #cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Prevents numerical errors
            angle = np.degrees(np.arccos(cos_theta))

            # Adjust for Quadrant II
            if Bx < 0:
                #OpenCV_dbprint(f"Quadrant II, {Bx, By}", (cx_black + 5, cy_black + 5), (255, 0, 0))
                angle = 180 - angle  # Reflect across vertical axis

            # adjust for Quadrant III
            if Bx < 0 and By < 0:
                #OpenCV_dbprint(f"Quadrant III, {Bx, By}", (cx_black + 5, cy_black + 5), (255, 0, 0))
                angle = 180 - angle + 180

            # adjust for Quadrant IV (use negative angles)
            if Bx > 0 > By:
                #OpenCV_dbprint(f"Quadrant IV, {Bx, By}", (cx_black + 5, cy_black + 5), (255, 0, 0))
                # use negative angles instead of positive angles for this quadrant
                angle = -angle

            # Draw indicators
            cv2.circle(frame, (cx_black, cy_black), 5, (0, 255, 0), -1)  # Black piece centroid
            cv2.line(frame, circle_center, (cx_black, cy_black), (255, 0, 0), 2)  # Line to black piece

            return angle, (cx_black, cy_black)
    return None, (None, None)


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
        circles = np.int16(np.around(circles))
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
previous_time = start_time
while True:
    # Check if 0.1 seconds have passed
    elapsed_time = time.time() - previous_time
    if elapsed_time < 0.1:
        continue

    previous_time = time.time()
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

        # ==== Frame Differencing for Motion Detection ====
        if prev_gray is None:
            prev_gray = gray.copy()
            continue
        if prev_gray is not None:
            # Compute absolute difference
            frame_diff = cv2.absdiff(prev_gray, gray)
            _, motion_mask = cv2.threshold(frame_diff, 35, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Find contours in motion mask
            motion_contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if motion_contours:
                largest_contour = max(motion_contours, key=cv2.contourArea)
                bounding_box = cv2.boundingRect(largest_contour)
                x_m, y_m, w_m, h_m = bounding_box
                x_m, y_m = int(x_m), int(y_m)
                x_m += roi_x1
                y_m += roi_y1
                bounding_box = (x_m, y_m, w_m, h_m)
                cv2.rectangle(frame, (x_m, y_m), (x_m + w_m, y_m + h_m), (0, 255, 255), 2)

                # Call function to detect black piece and calculate angle
                angle, (cx_black, cy_black) = find_black_piece_and_angle(frame, bounding_box, ball_center)

                # === Confidence estimation ===
                marker_area = cv2.contourArea(largest_contour) if largest_contour is not None else 0
                motion_score = np.sum(motion_mask) / 255
                dist = np.sqrt((cx_black - cx) ** 2 + (cy_black - cy) ** 2) if angle is not None else 0

                # Normalize and combine scores
                MAX_AREA = 200  # Adjust as needed
                MAX_MOTION = 100  # Adjust as needed
                MAX_DIST = 2 * radius  # Example, adjust based on setup

                conf_area_penalty = np.clip(marker_area / MAX_AREA, 0, 1)
                conf_motion = np.clip(motion_score / MAX_MOTION, 0, 1)
                inv_conf_motion = 1 - (conf_motion ** 2.5)  # Invert motion contribution

                # Distance confidence with proximity penalty
                if dist < radius:
                    conf_dist = (dist / radius) ** 2  # Quadratic penalty inside radius
                else:
                    conf_dist = np.clip((dist - radius) / (MAX_DIST - radius), 0, 1)  # Linear beyond

                # After obtaining the current angle and before updating confidence:
                if len(angle_values) >= 2 and angle is not None and angle_values[-2] is not None:
                    delta_angle = abs(angle - angle_values[-2])
                    delta_time = Time[-1] - Time[-2]
                    angular_velocity = delta_angle / delta_time
                else:
                    angular_velocity = 0.0

                max_angular_velocity = 100  # Adjust based on expected maximum
                angular_penalty = np.clip(angular_velocity / max_angular_velocity, 0, 1)

                # Combine all factors
                confidence_score = (
                        0.5 * inv_conf_motion +
                        0.3 * conf_dist +
                        0.2 * conf_area_penalty +
                        0.3 * angular_penalty
                        + 0.35
                )

                # Normalize confidence score to [0, 1]
                confidence_score = np.clip(confidence_score, 0.0, 1.0)

                # Update the confidence score label
                confidence_score_label.config(text=f"Confidence Score: {confidence_score:.2f}")

                #print(angle)
                # add angle data to live feed
                if angle:
                    angle_values.append(angle)  # Store angle for plotting
                    Time.append(time.time() - start_time)
                    UTC_time.append(datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3])
                    confidence_values.append(confidence_score)
                    cv2.putText(frame, f"Angle: {angle:.2f} degrees", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(frame, "TRACKING ACTIVE", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw a small circle at the centroid of the detected motion
            if bounding_box is not None:
                cx_m = bounding_box[0] + bounding_box[2] // 2
                cy_m = bounding_box[1] + bounding_box[3] // 2
                cv2.circle(frame, (cx_m, cy_m), 5, (0, 0, 255), -1)  # Centroid of motion

        # Store current frame as previous frame for next iteration
        prev_gray = gray.copy()

        # Draw cue ball circle on all feeds
        cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(gray, (cx, cy), 2, (0, 0, 255), 2)
        cv2.line(frame, ball_center, (cx + radius, cy), (255, 0, 0), 2)  # Horizontal line

        # Display frames
        if angle is None:
            cx_black, cy_black = 0, 0
            angle_values.append(np.nan)
            confidence_values.append(0.0)  # Append low confidence placeholder
            Time.append(time.time() - start_time)
            UTC_time.append(datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3])
            cv2.putText(frame, "Angle: NaN", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.line(frame, ball_center, (cx + radius, cy), (255, 0, 0), 2)  # Horizontal line
            cv2.putText(frame, "TRACKING INACTIVE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Live Feed", frame)
        cv2.imshow("Gray Feed", gray)
        # Display the motion mask
        if motion_contours:
            cv2.imshow("Optical Flow Motion", motion_mask)

        # display a constantly updating graph of angles based on the black piece
        # Set labels
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title("Live Angle Tracking")

        # Create animation
        #ani = FuncAnimation(fig, update, interval=100)  # Update every 100ms

    # Press 'q' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ax.scatter(Time, angle_values, c='blue', s=1)  # Scatter plot of angles

# save all of this to a csv file
data = {'UTC Time': UTC_time, 'Time': Time, 'Angle': angle_values, 'Confidence': confidence_values}
df = pd.DataFrame(data)
if direction == 90:
    df.to_csv(f'raw_data/90angle_data_{amps}A.csv', index=False)  # Save to CSV
else:
    df.to_csv(f'raw_data/angle_data_{amps}A.csv', index=False)  # Save to CSV

plt.show()  # Show graph window
