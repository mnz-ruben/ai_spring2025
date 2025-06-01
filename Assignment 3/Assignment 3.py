#!/usr/bin/env python3
"""
plant_disease_classifier.py

A complete CNN implementation using TensorFlow/Keras to identify plant leaf diseases.
Assumes you have a directory structure like:

    plantvillage/
        Pepper__bell___Bacterial_spot/
            image1.png
            image2.png
            ...
        Pepper__bell___healthy/
            ...
        Potato__Early_blight/
            ...
        Potato__healthy/
            ...
        Potato__Late_blight/
            ...
        Tomato__Target_Spot/
            ...
        Tomato__Tomato_mosaic_virus/
            ...
        Tomato__Tomato_YellowLeaf__Curl_Virus/
            ...
        Tomato__Bacterial_spot/
            ...
        Tomato__Early_blight/
            ...
        Tomato__healthy/
            ...
        Tomato__Late_blight/
            ...
        Tomato__Leaf_Mold/
            ...
        Tomato__Septoria_leaf_spot/
            ...
        Tomato__Spider_mites__two_spotted_spider_mite/
            ...
        (etc.)
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    # 1) Configure paths and parameters
    # Replace 'plantvillage' with the actual path if it's different
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plantvillage")
    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please ensure 'plantvillage2/' with subfolders of JPG images is in the same folder as this script.")
        return

    # Image size and batch size
    img_height = 224
    img_width = 224
    batch_size = 32

    # 2) Create training and validation datasets from directory
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected classes: {class_names}")
    print(f"Number of classes: {num_classes}")

    # 3) Performance optimizations (prefetch, caching)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 4) Build the CNN model
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 5) Train the model
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # 6) Save the trained model
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plant_disease_model.h5")
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    # 7) (Optional) Plot training history if running in an environment that supports plotting
    try:
        import matplotlib.pyplot as plt

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(epochs)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig("training_history.png")
        print("Saved training history plot as 'training_history.png'.")
    except ImportError:
        print("matplotlib not installed; skipping training history plot.")

if __name__ == "__main__":
    main()
