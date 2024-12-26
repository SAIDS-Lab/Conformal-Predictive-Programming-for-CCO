"""
In this file, we implement the optimal control problem with an emphasis on the robust evaluation.
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
hyperparameters = {"N": 100, "K": 80, "L": 200, "V": 1000, "delta": 0.2}


mean_1 = 0
var_1 = 0.1
mean_2 = 0
var_2 = 0.13

def generate_training_random_noise():
    return np.random.normal(loc = mean_1, scale = var_1, size = T)

def generate_testing_random_noise():
    return np.random.normal(loc = mean_2, scale = var_2, size = T)
     

def f(u, Y):
    y0 = np.array([0, 0, 0, 0])
    A = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    B = np.array([[0.5, 0], [1, 0], [0, 0.5], [0, 1]])
    ys = [y0]
    for t in range(T):
        y_new = A @ ys[-1] + B @ np.array([u[t, 0], u[t, 1]]) + Y[t]
        ys.append(y_new)
    yT = ys[-1]
    return ((yT[0] - z[0]) * (yT[0] - z[0]) + (yT[2] - z[1]) * (yT[2] - z[1])) ** 2 - 3((yT[0] - z[0]) ** 2) * ((yT[2] - z[1]) ** 2) - zeta


def f_value(u, Y):
    y0 = np.array([0, 0, 0, 0])
    A = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    B = np.array([[0.5, 0], [1, 0], [0, 0.5], [0, 1]])
    ys = [y0]
    for t in range(T):
        y_new = A @ ys[-1] + B @ np.array([u[t][0], u[t][1]]) + Y[t]
        ys.append(y_new)
    yT = ys[-1]
    return ((yT[0] - z[0]) * (yT[0] - z[0]) + (yT[2] - z[1]) * (yT[2] - z[1])) ** 2 - 3((yT[0] - z[0]) ** 2) * ((yT[2] - z[1]) ** 2) - zeta

gs = []
hs = []


def J(u):
    return sum(u[t, 0] * u[t, 0] + u[t, 1] * u[t, 1] for t in range(T))


def J_value(u):
    return sum(u[t][0] * u[t][0] + u[t][1] * u[t][1] for t in range(T))


def compute_divergence(mean_test, var_test, mean_train, var_train):
    # Generate a T x T identity matrix.
    I = np.identity(T)
    inv_cov_test = np.linalg.inv(var_test)
    trace_term = np.trace(np.dot(inv_cov_test, var_train)-I)
    diff_mean = mean_test - mean_train
    quad_form = np.dot(diff_mean.T, np.dot(inv_cov_test, diff_mean))
    log_det_ratio = np.log(np.linalg.det(np.dot(var_train, inv_cov_test)))
    kl_divergence = 0.5 * (trace_term + quad_form - log_det_ratio)
    print("the KL-divergence is:", kl_divergence)
    return kl_divergence


# Run the first step of the experiment.
results_step_1 = dict()
results_baseline_step_1 = dict()

print("Calculating Distribution Shift:")
# Diagonalize the covariance matrices.
var_1_cov = np.diag([var_1 for _ in range(T)])
var_2_cov = np.diag([var_2 for _ in range(T)])
found_epsilon = compute_divergence(np.array([mean_2 for _ in range(T)]), var_2_cov, np.array([mean_1 for _ in range(T)]), var_1_cov)

print("Evaluating with CPP-MIP:")
results_step_1["CPP-MIP"] = run_experiment_step_1("marginal", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], hyperparameters["beta"], generate_training_random_noise, generate_testing_random_noise, hs, gs, (T, 2), f, J, f_value, J_value, robust = True, epsilon = found_epsilon)
results_baseline_step_1["CPP-MIP"] = run_experiment_step_1("marginal", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], hyperparameters["beta"], generate_training_random_noise, generate_testing_random_noise, hs, gs, (T, 2), f, J, f_value, J_value, robust = False, epsilon = None)
print()

# Save the results for the first step of the experiment.
with open("case_studies_results/results_case_study_3/results_step_1_robust.json", "w") as file:
    json.dump(results_step_1, file)
with open("case_studies_results/results_case_study_3/results_baseline_step_1.json", "w") as file:
    json.dump(results_baseline_step_1, file)

# Run the second step of the experiment.
with open("case_studies_results/results_case_study_3/results_step_1_robust.json", "r") as file:
    results_step_1 = json.load(file)
with open("case_studies_results/results_case_study_3/results_baseline_step_1.json", "r") as file:
    results_baseline_step_1 = json.load(file)

# Run the second step of the experiment.
print("Performing the second step of the experiment with the specified calibration parameters.")
results_step_2 = dict()
results_step_2["CPP-MIP"] = run_experiment_step_2(results_step_1["CPP-MIP"], hyperparameters["L"], hyperparameters["Z"], hyperparameters["W"], hyperparameters["beta"], generate_testing_random_noise, f_value, robust = True, epsilon = found_epsilon)
results_baseline_step_2 = dict()
results_baseline_step_2["CPP-MIP"] = run_experiment_step_2(results_baseline_step_1["CPP-MIP"], hyperparameters["L"], hyperparameters["Z"], hyperparameters["W"], hyperparameters["beta"], generate_testing_random_noise, f_value, robust = False, epsilon = None)

# Save the results from the second step.
with open("case_studies_results/results_case_study_3/results_step_2_robust.json", "w") as file:
    json.dump(results_step_2, file)
with open("case_studies_results/results_case_study_3/results_baseline_step_2.json", "w") as file:
    json.dump(results_baseline_step_2, file)