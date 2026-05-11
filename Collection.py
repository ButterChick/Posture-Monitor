import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import csv
import os
import urllib.request

MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

CSV_FILE = "posture_landmarks.csv"
CURRENT_LABEL = "good"  # change to "bad" for second session

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(13):
            header += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]
        header.append("label")
        writer.writerow(header)

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=result_callback
)

count = 0
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

            # Draw joints manually
            h, w = frame.shape[:2]
            for p in lm:
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(frame, (cx, cy), 4, (180, 180, 180), -1)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                row = []
                for p in lm:
                    row += [p.x, p.y, p.z]
                row.append(CURRENT_LABEL)

                with open(CSV_FILE, "a", newline="") as f:
                    csv.writer(f).writerow(row)

                count += 1
                print(f"Saved sample {count} — label: {CURRENT_LABEL}")

        cv2.putText(frame, f"Label: {CURRENT_LABEL} | Saved: {count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press S to save, Q to quit",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Collect Landmarks", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Done. Total samples saved: {count}")