from __future__ import print_function
import os
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Load and process raw prediction CSV files
def load_and_process_raw_csv(file_path):
    data = pd.read_csv(file_path, header=None)
    data = np.asarray(data)
    data = np.delete(data, 0, 1)  # Remove the first column
    data = np.delete(data, 0, 0)  # Remove the first row
    return data  # Return raw predictions

# Paths to your prediction files and ground truth labels
pred_paths = [
    'Model_1_pred.csv',
    'Model_2_pred.csv',
    'Model_3_pred.csv',
    'Model_4_pred.csv',
    'Model_5_pred.csv'
]

# Start timing the process
tic = time.time()

# Load raw predictions
raw_preds = [load_and_process_raw_csv(path) for path in pred_paths]

# Load and process ground truth labels
labels = pd.read_csv('GT_labels.csv', header=None)
labels = np.asarray(labels)
labels = np.delete(labels, 0, 1)
labels = np.delete(labels, 0, 0)
labels = labels.ravel()

# Number of classes
n_classes = 5

# Calculate AROC-based weights for each classifier
def calculate_aroc_weights(preds, labels, n_classes):
    weights = []
    for pred in preds:
        aroc_scores = []
        for i in range(n_classes):
            # Calculate AROC for each class
            binary_labels = (labels == i).astype(int)
            aroc = roc_auc_score(binary_labels, pred[:, i])
            aroc_scores.append(aroc)
        # Average AROC across all classes
        Wk = np.mean(aroc_scores)
        weights.append(Wk)
    return np.array(weights)

# Calculate weights
weights = calculate_aroc_weights(raw_preds, labels, n_classes)
print("AROC-based Weights:", weights)

# Compute weighted average predictions
weighted_preds = np.average(raw_preds, axis=0, weights=weights)

# Convert weighted predictions to class labels
weighted_labels = np.argmax(weighted_preds, axis=1)

# Calculate accuracy for weighted averaging
accuracy_weighted = accuracy_score(labels, weighted_labels) * 100
print("Weighted Averaging Accuracy:", accuracy_weighted)

# End timing the process
toc = time.time()

# Calculate the confusion matrix for weighted averaging
confusion_weighted = confusion_matrix(labels, weighted_labels)
print("Confusion Matrix for Weighted Averaging:")
print(confusion_weighted)

# Print the execution time
print(f"Execution Time: {toc - tic:.4f} seconds")
