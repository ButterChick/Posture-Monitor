import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import os
import urllib.request
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


posture_model = tf.keras.models.load_model(
    "posture_lm.h5",
    compile=False
)

LANDMARK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

GOOD_CLASS_INDEX = 1

options = PoseLandmarkerOptions(
    base_options = python.BaseOptions(model_asset_path = MODEL_PATH),
    running_mode = RunningMode.IMAGE,
)

try:
    posture_model = tf.keras.models.load_model(
        "posture_lm.h5",
        compile=False
    )
    print("Posture model loaded")

    landmarker = PoseLandmarker.create_from_options(options)
    print("MediaPipe loaded")

except Exception as e:
    print("STARTUP ERROR:", e)
    raise e

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)

    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return JSONResponse({"status":"No person Found"})
    
    lm = result.pose_landmarks[0]

    if len(lm) < 13:
        return JSONResponse({"status": "partial"})
    row = []
    for i in LANDMARK_INDICES:
        p = lm[i]
        row += [p.x,p.y,p.z]

    input_vec = np.array(row,dtype=np.float32).reshape(1,39)
    prediction = posture_model.predict(input_vec,verbose=0)[0][0]

    good_confidence = float(prediction if GOOD_CLASS_INDEX == 1 else 1 - prediction)
    good_posture = good_confidence < 0.65

    return JSONResponse({
        "status" : "good" if good_posture else "bad"
    })
app.mount("/", StaticFiles(directory="static", html=True), name="static")