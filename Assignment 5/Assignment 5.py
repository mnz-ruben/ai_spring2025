#!/usr/bin/env python3
"""
handwriting_hmm.py

Handwritten digit recognition using Hidden Markov Models (HMM) on the MNIST dataset.

This script:
  1. Loads the MNIST digits dataset.
  2. Extracts simple column-wise pixel features to create sequences.
  3. Trains one Gaussian HMM per digit class (0-9) using a subset of training images.
  4. Evaluates on a subset of the test images and reports accuracy and a confusion matrix.

Requirements:
  pip install numpy scikit-learn hmmlearn tensorflow matplotlib
"""

import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from hmmlearn.hmm import GaussianHMM
import pickle

def extract_sequence(image):
    """
    Convert a 28x28 MNIST image into a sequence of 28 observations.
    Each observation is a 28-dimensional vector: the normalized pixel values of one column.
    """
    normalized = image.astype(np.float32) / 255.0
    sequence = normalized.T  # shape (28, 28)
    return sequence

def prepare_hmm_training_data(images, labels, digit, n_samples):
    """
    For a given digit (0-9), select up to n_samples images from 'images' with matching 'labels'.
    Return:
      - concatenated feature array of shape (n_samples*28, 28)
      - lengths array of length n_samples (each equal to 28)
    """
    idx = np.where(labels == digit)[0][:n_samples]
    sequences = [extract_sequence(images[i]) for i in idx]
    lengths = [seq.shape[0] for seq in sequences]  # each = 28
    concat = np.vstack(sequences)
    return concat, lengths

def train_hmms(X_train, y_train, n_train_per_class=200, n_states=8, n_iter=10):
    """
    Train one GaussianHMM per digit class using up to n_train_per_class training images.
    Returns a dict mapping digit -> trained HMM model.
    """
    hmms = {}
    for digit in range(10):
        X_digit, lengths = prepare_hmm_training_data(X_train, y_train, digit, n_train_per_class)
        model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=n_iter, verbose=False, random_state=42)
        model.fit(X_digit, lengths)
        hmms[digit] = model
        print(f"Trained HMM for digit {digit} ({len(lengths)} sequences, {n_states} states).")
    return hmms

def recognize(hmms, image):
    """
    Given a trained HMM dict and a single MNIST image, predict the digit by choosing the HMM
    with the highest log-likelihood for the image's sequence.
    """
    seq = extract_sequence(image)
    best_score = -np.inf
    best_digit = None
    for digit, model in hmms.items():
        try:
            score = model.score(seq)
        except:
            score = -np.inf
        if score > best_score:
            best_score = score
            best_digit = digit
    return best_digit

def evaluate(hmms, X_test, y_test, n_test_per_class=50):
    """
    Evaluate HMM models on up to n_test_per_class test images per digit.
    Returns overall accuracy and confusion matrix.
    """
    y_true = []
    y_pred = []
    for digit in range(10):
        idx = np.where(y_test == digit)[0][:n_test_per_class]
        for i in idx:
            predicted = recognize(hmms, X_test[i])
            y_true.append(digit)
            y_pred.append(predicted)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    return acc, cm

def plot_confusion_matrix(cm):
    """
    Displays a confusion matrix heatmap and saves it to 'confusion_matrix.png'.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion matrix to 'confusion_matrix.png'.")

def main():
    # 1) Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("Loaded MNIST:")
    print(f"  Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 2) Train HMMs (one per digit 0-9)
    n_train_per_class = 200   # adjust for speed/accuracy tradeoff
    n_states = 8              # number of hidden states per HMM
    n_iter = 10               # Baum-Welch iterations
    print("\nTraining HMMs...")
    hmms = train_hmms(X_train, y_train,
                      n_train_per_class=n_train_per_class,
                      n_states=n_states, n_iter=n_iter)

    # 3) Evaluate on test set
    n_test_per_class = 50
    print("\nEvaluating on test set...")
    accuracy, cm = evaluate(hmms, X_test, y_test, n_test_per_class=n_test_per_class)
    print(f"\nAccuracy on {n_test_per_class} test samples per class: {accuracy*100:.2f}%")

    # 4) Plot confusion matrix
    plot_confusion_matrix(cm)

    # 5) Save all HMMs to a pickle file
    with open("hmm_models.pkl", "wb") as f:
        pickle.dump(hmms, f)
    print("Saved HMM models to 'hmm_models.pkl'.")

if __name__ == "__main__":
    main()
