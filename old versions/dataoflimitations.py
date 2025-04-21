import cv2
import time
import numpy as np

# Try to find the max supported resolution
def get_max_resolution(cam):
    common_resolutions = [
        (3840, 2160),  # 4K
        (2560, 1440),  # QHD
        (1920, 1080),  # Full HD
        (1280, 720),   # HD
        (640, 480),    # VGA
    ]
    max_res = (0, 0)
    for width, height in common_resolutions:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == width and actual_h == height:
            max_res = (width, height)
            break  # Keep highest one that worked

    return max_res

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Get max resolution
max_width, max_height = get_max_resolution(cap)

# Font and variables
font = cv2.FONT_HERSHEY_SIMPLEX
prev_pos = None
prev_time = time.time()

# Stats tracking
frame_times = []
speeds = []
tracking_failures = 0
frame_count = 0
motion_blur_warnings = 0

print("Move an object in front of the camera at different speeds.")
print("Press 'q' to quit and get a report.")
print(f"Max Supported Resolution: {max_width} x {max_height}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    frame_count += 1
    curr_time = time.time()
    frame_times.append(curr_time)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            (x, y, w, h) = cv2.boundingRect(largest_contour)
            center = (x + w//2, y + h//2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            if prev_pos is not None:
                dt = curr_time - prev_time
                dx = center[0] - prev_pos[0]
                dy = center[1] - prev_pos[1]
                dist = (dx**2 + dy**2)**0.5
                speed = dist / dt
                speeds.append(speed)

                if speed > 1000:
                    motion_blur_warnings += 1
                    cv2.putText(frame, "Motion blur warning!", (10, 80), font, 0.6, (0, 0, 255), 2)

                cv2.putText(frame, f"Speed: {speed:.1f} px/s", (10, 50), font, 0.6, (255, 255, 0), 2)

            prev_pos = center
            prev_time = curr_time
        else:
            tracking_failures += 1
    else:
        tracking_failures += 1

    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS (Reported): {reported_fps:.1f}", (10, 20), font, 0.6, (255, 0, 0), 2)

    cv2.imshow('Motion Speed Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Analyze FPS
actual_fps = len(frame_times) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
dropped_frames = sum(
    1 for i in range(1, len(frame_times))
    if (frame_times[i] - frame_times[i-1]) > 1.5 / actual_fps
)

# Final report
print("\n--- CAMERA SOFTWARE LIMITATIONS REPORT ---")
print(f"Max Supported Resolution: {max_width} x {max_height}")
print(f"Reported FPS: {reported_fps:.1f}")
print(f"Actual FPS: {actual_fps:.1f}")
print(f"Dropped Frames (software/hardware lag): {dropped_frames}")
print(f"Total Tracking Failures (low contrast/blur): {tracking_failures}")
print(f"Motion Blur Warnings (likely blur/speed limit exceeded): {motion_blur_warnings}")

if speeds:
    print(f"Max Tracked Speed: {max(speeds):.2f} px/sec")
    print(f"Avg Tracked Speed: {sum(speeds)/len(speeds):.2f} px/sec")

# Limitations Diagnosis
print("\n--- DIAGNOSTICS ---")
if actual_fps < reported_fps * 0.9:
    print("⚠️ Frame rate bottleneck detected — software can't maintain advertised FPS.")
if dropped_frames > 0:
    print("⚠️ Frame drops detected — camera/USB/software may be lagging.")
if tracking_failures > frame_count * 0.1:
    print("⚠️ Frequent tracking loss — may be due to motion blur or poor lighting.")
if motion_blur_warnings > 0:
    print("⚠️ High-speed motion appears blurry — sensor/software can't resolve fast movement.")
if actual_fps < 20:
    print("⚠️ FPS too low for reliable motion tracking.")

print("\n✅ Test complete.")
