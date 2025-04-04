import numpy as np
import pandas as pd 
from scipy.optimize import differential_evolution
from grid import Gompertz 
# Load the predictions and ground truth labels
p1 = pd.read_csv('Model_1_pred.csv', header=None)
p2 = pd.read_csv('Model_2_pred.csv', header=None)
p3 = pd.read_csv('Model_3_pred.csv', header=None)
p4 = pd.read_csv('Model_4_pred.csv', header=None)
p5 = pd.read_csv('Model_5_pred.csv', header=None)
labels = pd.read_csv('GT_labels.csv', header=None)

# Remove unnecessary rows and columns
p1 = np.asarray(p1)[1:, 1:]
p2 = np.asarray(p2)[1:, 1:]
p3 = np.asarray(p3)[1:, 1:]
p4 = np.asarray(p4)[1:, 1:]
p5 = np.asarray(p5)[1:, 1:]
labels = np.asarray(labels)[1:, 1:]

# Define the evaluation function for differential evolution
def evaluate_de(params):
    top, alpha_1, alpha_2, alpha_3 = map(int, np.round(params))
    predictions = Gompertz(top, alpha_1, alpha_2, alpha_3, p1, p2, p3, p4, p5)
    predictions = np.squeeze(predictions)
    correct = np.sum(predictions == labels)
    total = labels.shape[0]
    accuracy = correct / total
    return -accuracy  # Minimize negative accuracy

# Define the bounds for each parameter
bounds = [(1, 5), (1, 10), (1, 10), (1, 10)]

# Run the differential evolution algorithm
result = differential_evolution(evaluate_de, bounds, strategy='best1bin', maxiter=100, popsize=100, tol=1e-6)

# Print the result
best_params = result.x
best_accuracy = -result.fun
print(f"Best parameters: {best_params}")
