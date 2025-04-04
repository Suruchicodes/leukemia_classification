Leukemia Classification Using Ensemble Learning
Overview
This project enhances leukemia classification accuracy by combining multiple model predictions using ensemble learning. It explores majority voting, Gompertz-based ranking, genetic algorithms, and differential evolution for optimization.

Key Components
Majority Voting (majority.py): Aggregates predictions from five models.
Gompertz-Based Ranking (grid.py): Uses a ranking function to refine predictions.
Differential Evolution (differential_evolution.py): Optimizes Gompertz parameters.
Genetic Algorithm (genetic_algorithm.py): Evolves optimal parameters through selection and mutation.
Data & Techniques
Input: Model predictions (Model_1_pred.csv to Model_5_pred.csv), ground truth (GT_labels.csv).
Methods: Majority voting, Gompertz function, differential evolution, genetic algorithm.
Metrics: Accuracy, confusion matrix, execution time.
Tools & Libraries
Python, NumPy, Pandas, Scikit-learn, DEAP, SciPy.
Usage
Clone repository & install dependencies:
git clone https://github.com/Suruchicodes/leukemia_clasification.git
cd Ensemble-Learning
pip install -r requirements.txt
Run scripts:
python majority.py
python grid.py
python differential_evolution.py
python genetic_algorithm.py
Applications
Applicable to medical classification problems and other machine learning tasks requiring ensemble optimization.
##Note- alpha_1 = alpha, alpha_2 = beta, alpha_3 = gamma.
