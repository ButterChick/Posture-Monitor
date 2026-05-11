import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import os
import urllib.request
from Alert import Posture_Alert
import time

MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

posture_model = tf.keras.models.load_model("posture_lm.h5")

GOOD_CLASS_INDEX = 0

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = PoseLandmarkerOptions(
    base_options = python.BaseOptions(model_asset_path = MODEL_PATH),
    running_mode = RunningMode.LIVE_STREAM,
    result_callback = result_callback
)

cam = cv2.VideoCapture(0)
alert = Posture_Alert(frequency=1000,duration=400)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened():
        ret,frame = cam.read()
        if not ret:
            break

        timestamp = int(cam.get(cv2.CAP_PROP_POS_MSEC))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)
        landmarker.detect_async(mp_image,timestamp)

        if latest_result and latest_result.pose_landmarks:
            lm = latest_result.pose_landmarks[0]
            LANDMARK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            row = []
            for i in LANDMARK_INDICES:
                p = lm[i]
                row += [p.x,p.y,p.z]

            input_vec = np.array(row, dtype=np.float32).reshape(1,39)
            prediction = posture_model.predict(input_vec, verbose = 0)[0][0]

            good_confidence = prediction if GOOD_CLASS_INDEX == 1 else 1 - prediction
            good_posture = good_confidence > 0.65
            alert.update(good_posture)

            color = (0, 255, 0) if good_posture else (0, 0, 255)
            alert.update(good_posture)
            duration = alert.bad_duration()
            status = "Good posture" if good_posture else f"Fix posture! ({duration}s)"

            cv2.putText(frame, status, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("Posture Monitor", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
