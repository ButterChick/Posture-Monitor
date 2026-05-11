import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import urllib.request
import os

MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=result_callback
)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_result and latest_result.pose_landmarks:
            lm = latest_result.pose_landmarks[0]

            # Landmark indices: 7=left ear, 11=left shoulder, 23=left hip, 12=right shoulder
            left_ear      = (lm[7].x,  lm[7].y)
            left_shoulder = (lm[11].x, lm[11].y)
            left_hip      = (lm[23].x, lm[23].y)
            right_shoulder= (lm[12].x, lm[12].y)

            neck_angle    = calculate_angle(left_ear, left_shoulder, left_hip)
            shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1]) * 100

            good_posture = (160 < neck_angle < 200) and (shoulder_tilt < 3)
            color = (0, 255, 0) if good_posture else (0, 0, 255)
            status = "Good posture" if good_posture else "Fix posture!"

            # Draw landmarks manually
            h, w = frame.shape[:2]
            for p in lm:
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(frame, (cx, cy), 4, color, -1)

            cv2.putText(frame, status, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f"Neck angle: {neck_angle:.1f}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Shoulder tilt: {shoulder_tilt:.1f}%", (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Posture Monitor", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()