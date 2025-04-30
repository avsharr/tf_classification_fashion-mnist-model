# tf_classification_fashion-mnist-model
"""
🧠 Fashion MNIST Classifier — Neural Network for Clothing Recognition

This project is a simple neural network built using TensorFlow/Keras, designed to classify images of clothing from the Fashion MNIST dataset.
The model is trained to distinguish 10 categories: T-shirts, sneakers, coats, and more.

📦 Technologies Used:
Python
TensorFlow / Keras
Neural network (Sequential API)
Prediction visualization and confusion matrix
Image preprocessing (normalization)
Learning Rate Scheduler Callback

📊 Model Architecture:
Flatten (converts the 28x28 image into a vector)
Dense (hidden layer with 4 neurons, ReLU activation)
Dense (another hidden layer with 4 neurons)
Dense (output layer with 10 neurons and softmax)
