from __future__ import print_function
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# Load and preprocess predictions
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path, header=None)
    data = np.asarray(data)
    data = np.delete(data, 0, axis=1)  # Remove the first column
    data = np.delete(data, 0, axis=0)  # Remove the first row
    return np.argmax(data, axis=1)

# Paths to prediction files
file_paths = [
    'Model_1_pred.csv',
    'Model_2_pred.csv',
    'Model_3_pred.csv',
    'Model_4_pred.csv',
    'Model_5_pred.csv'
]

# Start timing
tic = time.time()

# Load predictions
preds = [load_and_preprocess(fp) for fp in file_paths]

# Load and preprocess ground truth labels
labels = pd.read_csv('GT_labels.csv', header=None)
labels = np.asarray(labels)
labels = np.delete(labels, 0, axis=1)
labels = np.delete(labels, 0, axis=0).ravel()

# Perform majority voting
pred = [tuple(p) for p in preds]
vote_counts = Counter(pred)
majority_vote = np.array(vote_counts.most_common(1)[0][0]).reshape(-1, 1)

# Calculate the accuracy for majority voting
accuracy_majority = np.mean(majority_vote.ravel() == labels) * 100
print(f"Majority Voting Accuracy: {accuracy_majority:.2f}%")

# Stop timing
toc = time.time()

# Calculate the confusion matrix
confusion = confusion_matrix(labels, majority_vote.ravel())
print("Confusion Matrix:\n", confusion)

# Print the execution time
print(f"Execution Time: {toc - tic:.4f} seconds")
