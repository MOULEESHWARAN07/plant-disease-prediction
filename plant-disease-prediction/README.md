# 🌱 Plant Disease Prediction Using CNN

A beginner-friendly Machine Learning / Deep Learning project that classifies tomato leaf images into three classes:

1. Healthy
2. Early Blight
3. Late Blight

The project uses a small CNN built with TensorFlow/Keras and provides a simple Streamlit web application for prediction.

> **Important:** This is an educational image-classification project. A prediction from this model should not be treated as a professional agricultural diagnosis.

---

## 🧠 What You Learn

- Image dataset organization
- Image preprocessing
- Train / validation / test split
- CNN basics
- Model training
- Accuracy and loss graphs
- Classification report
- Confusion matrix
- Saving and loading a trained model
- Building a simple Streamlit application

---

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit

---

## 📊 Dataset

This project uses a **3-class subset of the PlantVillage dataset**.

The PlantVillage dataset contains images of healthy and diseased plant leaves. The tomato portion includes classes such as `Tomato___healthy`, `Tomato___Early_blight`, and `Tomato___Late_blight`.

Dataset sources:

- https://github.com/spMohanty/PlantVillage-Dataset
- https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

### Why the dataset is not included

The complete dataset is large, so it is intentionally not stored in this GitHub project. Download the dataset separately and place the required folders under `data/raw/`.

---

## 📁 Required Dataset Structure

After downloading PlantVillage, put these three folders inside:

```text
data/
└── raw/
    ├── Tomato___healthy/
    ├── Tomato___Early_blight/
    └── Tomato___Late_blight/
```

The folders should contain image files.

---

## 🚀 Installation

Open a terminal in the project folder.

### 1. Create a virtual environment (optional but recommended)

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 1️⃣ Prepare the Dataset

Run:

```bash
python prepare_dataset.py
```

The script creates:

```text
data/processed/
├── train/
├── val/
└── test/
```

The split is approximately:

- 70% training
- 15% validation
- 15% testing

---

## 2️⃣ Train the CNN

Run:

```bash
python train_model.py
```

The model trains for 10 epochs.

After training, you will get:

```text
models/
├── plant_disease_model.keras
└── class_names.json
```

It also creates:

```text
training_accuracy.png
training_loss.png
```

---

## 3️⃣ Test One Image from the Terminal

Run:

```bash
python predict.py
```

Example:

```text
Enter image path: sample_images/leaf.jpg

Prediction
----------
Disease/Class : early_blight
Confidence    : 92.35%
```

---

## 4️⃣ Run the Web Application

After training:

```bash
streamlit run app.py
```

A browser window will open.

Upload a tomato leaf image and click **Predict**.

---

## 📓 Jupyter Notebook

The project also includes:

```text
notebooks/plant_disease_prediction.ipynb
```

The notebook follows the same workflow and is useful for explaining the project during interviews.

---

## 🏗️ Project Structure

```text
plant-disease-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│
├── notebooks/
│   └── plant_disease_prediction.ipynb
│
├── sample_images/
│
├── prepare_dataset.py
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧩 CNN Architecture

The model is intentionally simple:

```text
Input Image (128 × 128 × 3)
          ↓
      Rescaling
          ↓
    Conv2D (16)
          ↓
    MaxPooling
          ↓
    Conv2D (32)
          ↓
    MaxPooling
          ↓
    Conv2D (64)
          ↓
    MaxPooling
          ↓
       Flatten
          ↓
     Dense (64)
          ↓
      Dropout
          ↓
     Dense (3)
          ↓
     Prediction
```

---

## 📈 Evaluation

The project calculates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

It also plots:

- Training vs Validation Accuracy
- Training vs Validation Loss

---

## 💬 Interview Explanation

### 30-second version

> "I developed a basic plant disease classification system using a Convolutional Neural Network. I used a subset of the PlantVillage dataset containing healthy, early blight, and late blight tomato leaf images. I organized the images into training, validation, and testing sets, resized the images to 128 by 128 pixels, trained a CNN using TensorFlow and evaluated it using accuracy, precision, recall and a confusion matrix. Finally, I built a Streamlit application where users can upload a leaf image and get a predicted class with confidence."

### Why CNN?

CNNs are useful for image classification because convolution layers can learn visual patterns such as edges, shapes, textures, and higher-level image features.

### Why resize images?

Images can have different dimensions. Resizing them to 128 × 128 gives the neural network a consistent input size and reduces computation.

### Why train/validation/test?

- **Training:** learns model parameters
- **Validation:** checks performance during training
- **Testing:** final evaluation on unseen data

---

## ⚠️ Limitation

This beginner project is trained on a limited set of classes and a controlled image dataset. Real-world field images can have different lighting, backgrounds, camera quality, leaf positions, and disease severity. Therefore, the model should not be considered a production agricultural diagnostic system.

---

## 📚 Dataset Reference

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7, 1419.
