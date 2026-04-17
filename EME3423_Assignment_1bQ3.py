import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320, 240))
    top_left = frame
    top_right = cv2.flip(frame, 1)
    bottom_left = cv2.flip(frame, 0)
    bottom_right = cv2.flip(frame, -1)

    top_row = np.concatenate((top_left, top_right), axis=1)
    bottom_row = np.concatenate((bottom_left, bottom_right), axis=1)
    grid = np.concatenate((top_row, bottom_row), axis=0)

    cv2.imshow("Multiple outputs", grid)

    if cv2.waitKey(1) != -1:
        break

cap.release()
cv2.destroyAllWindows()
