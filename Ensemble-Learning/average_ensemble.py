from __future__ import print_function
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import mode
import time

# Load predictions and ground truth labels
def load_and_process_csv(file_path):
    data = pd.read_csv(file_path, header=None)
    data = np.asarray(data)
    data = np.delete(data, 0, 1)  # Remove the first column
    data = np.delete(data, 0, 0)  # Remove the first row
    return np.argmax(data, axis=1)  # Return class predictions

# Paths to your prediction files and ground truth labels
pred_paths = [
    'Model_1_pred.csv',
    'Model_2_pred.csv',
    'Model_3_pred.csv',
    'Model_4_pred.csv',
    'Model_5_pred.csv'
]
# Start timing
tic = time.time()

# Load predictions
preds = [load_and_process_csv(path) for path in pred_paths]

# Load and process ground truth labels
labels = pd.read_csv('GT_labels.csv', header=None)
labels = np.asarray(labels)
labels = np.delete(labels, 0, 1)
labels = np.delete(labels, 0, 0)
labels = labels.ravel()

# Perform majority voting
all_preds = np.vstack(preds)
majority_vote = mode(all_preds, axis=0).mode.ravel()

# Perform ensemble averaging (soft voting)
def load_and_process_raw_csv(file_path):
    data = pd.read_csv(file_path, header=None)
    data = np.asarray(data)
    data = np.delete(data, 0, 1)  # Remove the first column
    data = np.delete(data, 0, 0)  # Remove the first row
    return data  # Return raw predictions (not class labels)

# Load raw predictions for ensemble averaging
raw_preds = [load_and_process_raw_csv(path) for path in pred_paths]

# Calculate the ensemble average
ensemble_predictions = np.mean(raw_preds, axis=0)

# Convert ensemble average predictions to class labels
ensemble_labels = np.argmax(ensemble_predictions, axis=1)

# Calculate accuracy for ensemble averaging
accuracy_ensemble = accuracy_score(labels, ensemble_labels) * 100
print("Ensemble Averaging Accuracy:", accuracy_ensemble)

# Stop timing
toc = time.time()

# Calculate the confusion matrix for ensemble averaging
confusion_ensemble = confusion_matrix(labels, ensemble_labels)
print("Confusion Matrix for Average Ensemble:")
print(confusion_ensemble)

# Print the execution time
print(f"Execution Time: {toc - tic:.4f} seconds")
