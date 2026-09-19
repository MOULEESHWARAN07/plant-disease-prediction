from pathlib import Path
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

CLASS_NAMES = ["early_blight", "healthy", "late_blight"]

# -----------------------------
# Load datasets
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    class_names=CLASS_NAMES,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=CLASS_NAMES,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=CLASS_NAMES,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -----------------------------
# Build a beginner CNN
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# -----------------------------
# Save model and class names
# -----------------------------
model.save(MODEL_DIR / "plant_disease_model.keras")

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(CLASS_NAMES, f, indent=2)

print("\nModel saved to models/plant_disease_model.keras")

# -----------------------------
# Accuracy / loss graphs
# -----------------------------
plt.figure(figsize=(7, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_accuracy.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.show()

# -----------------------------
# Test evaluation
# -----------------------------
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Predictions for classification report
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
