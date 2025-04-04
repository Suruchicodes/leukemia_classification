import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

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

# Define the Gompertz function here
def Gompertz(top, alpha_1, alpha_2, alpha_3, *argv):
    # Ensure all values and calculations are real numbers
    top = int(round(top))  # Ensure top is an integer
    top = max(1, min(top, 5))  # Ensure top is within the range [1, 5]
    alpha_1 = abs(alpha_1)  # Ensure alpha_1 is non-negative
    alpha_2 = abs(alpha_2)  # Ensure alpha_2 is non-negative
    alpha_3 = abs(alpha_3)  # Ensure alpha_3 is non-negative

    if len(argv) < 5:
        raise ValueError("Insufficient arguments for Gompertz function")

    CF = np.array(argv)
    R_L = 1 - alpha_1 * np.exp(-alpha_2 * np.exp(-alpha_3 * CF))
    K_L = R_L.copy()

    # Apply penalties (Gompertz logic)
    G_zero = 1 - alpha_1 * np.exp(-alpha_2 * np.exp(-alpha_3 * 0))
    for i in range(CF.shape[0]):
        for j in range(CF.shape[1]):
            for k in range(top):
                idx = np.argpartition(R_L[i, j], -k-1)[-k-1]
                if R_L[i, j, idx] < G_zero:
                    K_L[i, j, idx] = G_zero

    RS = np.sum(K_L, axis=0)
    CFS = 1 - np.sum(CF, axis=0) / len(argv)
    FS = RS * CFS
    predictions = np.argmin(FS, axis=1)
    return predictions

# Define the evaluation function
def evaluate(individual):
    top, alpha_1, alpha_2, alpha_3 = individual
    predictions = Gompertz(top, alpha_1, alpha_2, alpha_3, p1, p2, p3, p4, p5)
    predictions = np.squeeze(predictions)
    correct = np.sum(predictions == labels)
    total = labels.shape[0]
    accuracy = correct / total
    return accuracy,

# Define the genetic algorithm parameters
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("attr_top", np.random.randint, 1, 6)
toolbox.register("attr_alpha", np.random.uniform, 1, 10)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_top, toolbox.attr_alpha, toolbox.attr_alpha, toolbox.attr_alpha), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=[3, 5, 5, 5], sigma=[1, 1, 1, 1], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Set parameters for the algorithm
population = toolbox.population(n=250)
ngen = 100
cxpb = 0.5
mutpb = 0.2

# Run the genetic algorithm
algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

# Extract and print the best individual
best_ind = tools.selBest(population, 1)[0]
print(f"Best individual is {best_ind}")
