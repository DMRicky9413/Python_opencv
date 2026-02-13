import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    original = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(frame, 100, 200)

    cv2.imshow("Webcam", original)
    cv2.imshow("Webcam Gaussian Blur", gaussian_blur)
    cv2.imshow("Webcam HSV", hsv)
    cv2.imshow("Webcam Canny Edge", edges)

    if cv2.waitKey(1) != -1:
        break
cap.release()
cv2.destroyAllWindows()

