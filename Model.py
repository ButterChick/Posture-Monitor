import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

CSV_File = "posture_landmarks.csv"

df = pd.read_csv(CSV_File)

x = df.drop("label", axis=1).values.astype(np.float32)
y_raw = df["label"].values

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw).astype(np.float32)
print(f"Label encoding: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state= 55, stratify=y
)

model = models.Sequential([
    layers.Input(shape=(39,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(rate=0.2),
    layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer = "adam",
    loss='binary_crossentropy',
    metrics = ['accuracy']
)

model.summary()

history = model.fit(
    x_train, y_train,
    validation_data = (x_val, y_val),
    epochs = 1000,
    batch_size = 16,
    verbose = 1
)

model.save("posture_lm.h5")
print(f"\nModel saved to posture_lm.h5")
print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.2%}")
