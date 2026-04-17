import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

count = 0
stage = "up"
angle_history = []

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Can't find camera.")
    exit()

WINDOW_NAME = "Push-Up Counter"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)


def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def smooth_angle(angle, history, window_size=3):
    history.append(angle)
    if len(history) > window_size:
        history.pop(0)
    return np.mean(history)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("skip")
        continue

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
        )


        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]

        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]


        l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)


        avg_angle = (l_angle + r_angle) / 2.0
        smooth_avg = smooth_angle(avg_angle, angle_history, window_size=3)


        cv2.putText(frame, f"L-Angle: {int(l_angle)}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"R-Angle: {int(r_angle)}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if smooth_avg > 160 and stage == "down":
            stage = "up"
            count += 1
            print(f"✔️ count: {count}")
        elif smooth_avg < 80 and stage == "up":
            stage = "down"

        cv2.putText(frame, f"Count: {count}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Status: {stage.upper()}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

        bar_length = int(np.interp(smooth_avg, [60, 170], [50, 300]))
        bar_length = np.clip(bar_length, 50, 300)
        cv2.rectangle(frame, (frame.shape[1] - 340, 30), (frame.shape[1] - 40, 70), (100, 100, 100), -1)
        cv2.rectangle(frame, (frame.shape[1] - 340, 30), (frame.shape[1] - 340 + bar_length, 70), (0, 255, 0), -1)
        cv2.putText(frame, "Angle", (frame.shape[1] - 380, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.circle(frame, tuple(np.array(l_elbow, dtype=int)), 6, (0, 0, 255), -1)
        cv2.circle(frame, tuple(np.array(r_elbow, dtype=int)), 6, (0, 0, 255), -1)

    else:

        cv2.putText(frame, "No body detected. Please stand in front of camera.", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "Press 'R' to reset | 'Q' to quit", (30, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow(WINDOW_NAME, frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(f"Total count: {count}")
        break
    elif key == ord('r'):
        count = 0
        stage = "up"
        angle_history.clear()
        print("Reset counter")

cap.release()
cv2.destroyAllWindows()
pose.close()