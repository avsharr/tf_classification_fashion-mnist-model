import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix
import itertools

# Load and normalize Fashion MNIST dataset
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
train_data_norm = train_data / 255.0
test_data_norm = test_data / 255.0

# Set random seed
tf.random.set_seed(42)

# Define class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Define callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Train the model
history = model.fit(train_data_norm, train_labels,
                    epochs=20,
                    validation_data=(test_data_norm, test_labels),
                    callbacks=[reduce_lr, early_stop])

# Predict class probabilities
y_probs = model.predict(test_data_norm)
y_preds = y_probs.argmax(axis=1)

# Function to create confusion matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    nb_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(nb_classes)

    ax.set(title="Confusion Matrix",
           xlabel="Predicted Label",
           ylabel="True Label",
           xticks=np.arange(nb_classes),
           yticks=np.arange(nb_classes),
           xticklabels=labels,
           yticklabels=labels)

    threshold = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)

    plt.show()

# Plot confusion matrix
make_confusion_matrix(y_true=test_labels, 
                      y_pred=y_preds,
                      classes=class_names,
                      figsize=(12, 12),
                      text_size=10)

# Function to plot a random image and prediction
def plot_random_image(model, images, true_labels, classes):
    i = random.randint(0, len(images) - 1)
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28, 28))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]

    plt.imshow(target_image, cmap=plt.cm.binary)

    color = "green" if pred_label == true_label else "red"
    plt.xlabel(f"Pred: {pred_label} ({100*tf.reduce_max(pred_probs):.0f}%) | True: {true_label}",
               color=color)
    plt.title("Random Prediction")
    plt.show()

# Show random prediction example
plot_random_image(model, test_data, test_labels, class_names)

