import cv2
import numpy as np

# Load the pre-trained face detector to test OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# we need to explicitly avoid the Continuity iphone camera and use the Macbook's camera
cap = cv2.VideoCapture(0)

# check if hte camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret,frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use edge detection to find contours
    edges = cv2.Canny(blurred, 50, 150)

    '''# Find faces
    faces = face_cascade.detectMultiScale(blurred, 1.1, 5)

    # draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)'''

    '''# Find circles (potential cue ball detection)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                param1=50, param2=30, minRadius=20, maxRadius=100)

    # Draw detected circles
    if circles is not None:
        circles = int(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw circle
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)  # Draw center
    '''
    # Display the result
    cv2.imshow('Detected Circles', frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
