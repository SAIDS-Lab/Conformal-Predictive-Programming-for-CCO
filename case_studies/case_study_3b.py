"""
In this file, we implement the optimal control problem with an emphasis on the mondrian evaluation.
"""

# Import necessary modules.
import numpy as np
from evaluate import run_experiment_step_1, run_experiment_step_2
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import configuration as config
import random
import numpy as np
import math

# Experimental setting:
np.random.seed(config.config_seed)
random.seed(config.config_seed)
T = 5
z = (5, 5)
zeta = 1
hyperparameters = {"N": 200, "K": 60, "L": 200, "V": 1000, "delta": 0.1, "beta": None, "Z": None, "W": None}


mean = 0
var = 0.01

def generate_training_random_noise():
    return np.random.normal(loc = mean, scale = var, size = (T, 4)).tolist()

def generate_testing_random_noise():
    return np.random.normal(loc = mean, scale = var, size = (T, 4)).tolist()
     

def f(u, Y):
    y0 = np.array([0, 0, 0, 0])
    A = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    B = np.array([[0.5, 0], [1, 0], [0, 0.5], [0, 1]])
    ys = [y0]
    for t in range(T):
        y_new = A @ ys[-1] + B @ np.array([u[t, 0], u[t, 1]]) + Y[t]
        ys.append(y_new)
    yT = ys[-1]
    return (yT[0] - z[0]) * (yT[0] - z[0]) + (yT[2] - z[1]) * (yT[2] - z[1]) - zeta


def f_value(u, Y):
    y0 = np.array([0, 0, 0, 0])
    A = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    B = np.array([[0.5, 0], [1, 0], [0, 0.5], [0, 1]])
    ys = [y0]
    for t in range(T):
        y_new = A @ ys[-1] + B @ np.array([u[t][0], u[t][1]]) + Y[t]
        ys.append(y_new)
    yT = ys[-1]
    return (yT[0] - z[0]) * (yT[0] - z[0]) + (yT[2] - z[1]) * (yT[2] - z[1]) - zeta

gs = []
hs = []


def J(u):
    return sum(u[t, 0] * u[t, 0] + u[t, 1] * u[t, 1] for t in range(T))


def J_value(u):
    return sum(u[t][0] * u[t][0] + u[t][1] * u[t][1] for t in range(T))

print("Evaluating with CPP-MIP:")
results_step_1 = dict()
results_step_1["CPP-MIP"] = run_experiment_step_1("marginal", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], hyperparameters["beta"], generate_training_random_noise, generate_testing_random_noise, hs, gs, (T, 2), f, J, f_value, J_value)
print()

# Save the results from the first step.
with open("case_studies_results/results_case_study_3/results_step_1_mondrian.json", "w") as file:
    json.dump(results_step_1, file)

# Load the results from the first step.
with open("case_studies_results/results_case_study_3/results_step_1_mondrian.json", "r") as file:
    results_step_1 = json.load(file)

# Write the function for is_mondrian_test_group.
def is_mondrian_test_group_case_3(Y):
    return np.any([abs(value) > 0.005 for value in np.array(Y).flatten()])

# Run the second step of the experiment.
print("Performing the second step of the experiment with the specified calibration parameters.")
results_step_2 = dict()
results_step_2["CPP-MIP"] = run_experiment_step_2(results_step_1["CPP-MIP"], hyperparameters["L"], hyperparameters["Z"], hyperparameters["W"], hyperparameters["beta"], generate_testing_random_noise, f_value, mondrian = True, is_mondrian_test_group=is_mondrian_test_group_case_3)

# Save the results from the second step.
with open("case_studies_results/results_case_study_3/results_step_2_mondrian.json", "w") as file:
    json.dump(results_step_2, file)