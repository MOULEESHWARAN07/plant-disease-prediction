# Interview Notes

## Project title
Plant Disease Prediction Using CNN

## Problem
Classify tomato leaf images into healthy, early blight, and late blight.

## Input
A tomato leaf image.

## Output
Predicted class and confidence.

## Algorithm
Convolutional Neural Network (CNN).

## Main steps
1. Collect/download PlantVillage data.
2. Select three tomato classes.
3. Split into train, validation, and test.
4. Resize images to 128x128.
5. Normalize pixel values using a Rescaling layer.
6. Train a CNN.
7. Evaluate using accuracy and classification metrics.
8. Save the model.
9. Create a Streamlit interface.

## Important questions
Q: Why CNN?
A: CNNs are designed to learn spatial patterns in images.

Q: Why resize?
A: Neural networks need a consistent input shape and smaller images reduce computation.

Q: What is validation data?
A: Data used to monitor model performance during training.

Q: What is test data?
A: Unseen data used for final evaluation.

Q: What does softmax do?
A: It converts the final outputs into class probabilities.

Q: What is overfitting?
A: When a model performs well on training data but poorly on unseen data.

Q: How can you improve this project?
A: Use data augmentation, transfer learning, more diverse field images, better class balancing, and hyperparameter tuning.
