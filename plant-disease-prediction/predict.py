from pathlib import Path
import json
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import tensorflow as tf

MODEL_PATH = Path("models/plant_disease_model.keras")
CLASS_PATH = Path("models/class_names.json")
IMAGE_SIZE = (128, 128)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

image_path = input("Enter image path: ").strip()

if not Path(image_path).exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

image = Image.open(image_path).convert("RGB")
image = image.resize(IMAGE_SIZE)

image_array = np.array(image, dtype=np.float32)
image_array = np.expand_dims(image_array, axis=0)

probabilities = model.predict(image_array, verbose=0)[0]
index = int(np.argmax(probabilities))

print("\nPrediction")
print("----------")
print(f"Disease/Class : {class_names[index]}")
print(f"Confidence    : {probabilities[index] * 100:.2f}%")
