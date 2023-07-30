import os
from sklearn.model_selection import cross_val_score
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# load the training dataset
training = pd.read_csv('new_oneHot_training.csv')

# Only have Waveform_type, Charge_per_phase, Charge_density, daily_accumulated_charge
filtered_columns = ['Waveform_type_biphasic_asymmetric','Waveform_type_biphasic_balanced', 'Waveform_type_biphasic_capacitive', 'Waveform_type_monophasic', 'GSA', 'Pulse_width', 'Frequency', 'Current', 'Voltage', 'Charge_per_phase', 'Charge_density', 'Current_density', 'stim_on', 'stim_total_day', 'daily_pulses', 'daily_accumulated_charge']

# Total columns
X_train = training[filtered_columns]
y_train = training['Avg_Damage_binary']

# list of activation functions to try
activations = ['relu', 'tanh', 'logistic']

# list of solvers to try
solvers = ['lbfgs', 'adam', 'sgd'] # this converges faster for small datasets like ours

hidden_layer_sizes = [(i, j) for i in range(1, 30) for j in range(1, 30)]

# prepare cross-validation
kf = KFold(n_splits=10)

# list of hyperparameters to try
param_grid = {
    'hidden_layer_sizes': hidden_layer_sizes,
    'activation': activations,
    'solver': solvers,
    'max_iter': [500]
}

# Define the MLP classifier
mlp = MLPClassifier(random_state=42)

results = pd.DataFrame(columns=["layer1_neurons", "layer2_neurons", "mean_accuracy", "std_accuracy"])

# Manually implement grid search
total_iterations = len(param_grid['hidden_layer_sizes']) * len(param_grid['activation']) * len(param_grid['solver']) * len(param_grid['max_iter'])
pbar = tqdm(total=total_iterations, ncols=80)

# Manually implement grid search
for hidden_layer in param_grid['hidden_layer_sizes']:
    for activation in param_grid['activation']:
        for solver in param_grid['solver']:
            for max_iter in param_grid['max_iter']:
                
                # Update hyperparameters
                mlp.set_params(hidden_layer_sizes=hidden_layer, activation=activation, solver=solver, max_iter=max_iter)

                # Perform cross-validation and compute mean and standard deviation of accuracy
                cv_scores = cross_val_score(mlp, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1)
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                print(f"MLP-{hidden_layer[0]}-{hidden_layer[1]} mean: {mean_score} std: {std_score}")

                # Only save the model if it meets the condition
                if mean_score > 0.75 and std_score < 0.10:
                    filename = f"C:\\Path_to_folder\\MLP-{hidden_layer[0]}-{hidden_layer[1]}-{activation}-{solver}.pkl"
                    # Concat result to results dataframe
                    joblib.dump(mlp, filename)

                result = [hidden_layer[0], hidden_layer[1], activation, solver, mean_score, std_score]
                results = pd.concat([results, pd.DataFrame([result], columns=["layer1_neurons", "layer2_neurons", "activation", "solver", "mean_accuracy", "std_accuracy"])])

                pbar.update(1)

# Save results to csv

results.to_csv('mlp_cv.csv', index=False)

pbar.close()