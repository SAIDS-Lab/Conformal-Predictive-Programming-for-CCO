"""
In this file, we implement the multiagent resource distribution problem.
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
import math

# Experimental setting:
np.random.seed(config.config_seed)
random.seed(config.config_seed)
hyperparameters = {"N": 100, "K": 80, "L": 200, "V": 1000, "delta": 0.1, "beta": None}

def generate_random_noise():
    return np.random.lognormal(mean=0, sigma=0.5, size=3)

A = np.array([[3, 12, 2], [10, 3, 5], [5, 3, 15]])
N_cons = 3

def f0(x, Y):
    return Y[0] - A[0][0]*x[0] + A[0][1]*x[1] + A[0][2]*x[2]


def f1(x, Y):
    return Y[1] - A[1][0]*x[0] + A[1][1]*x[1] + A[1][2]*x[2]


def f2(x, Y):
    return Y[2] - A[2][0]*x[0] + A[2][1]*x[1] + A[2][2]*x[2]

f = [f0, f1, f2]

def J(x):
    c = np.array([1, 1, 1])
    J = sum(c[i]*x[i] for i in range(N_cons))
    return J

hs = [lambda x: 0 - x[0], lambda x: 0 - x[1], lambda x: 0 - x[2]]
gs = []

print("Evaluating with CPP-MIP:")
results_step_1 = dict()
results_step_1["union"] = run_experiment_step_1("marginal", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], hyperparameters["beta"], generate_random_noise, generate_random_noise, hs, gs, 3, f, J, f, J, joint_method = "union")
results_step_1["max"] = run_experiment_step_1("marginal", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], hyperparameters["beta"], generate_random_noise, generate_random_noise, hs, gs, 3, f, J, f, J, joint_method = "max")
print()

# Save the results for the first step of the experiment.
with open("case_studies_results/results_case_study_4/results_step_1.json", "w") as file:
    json.dump(results_step_1, file)

# Run the second step of the experiment.
with open("case_studies_results/results_case_study_4/results_step_1.json", "r") as file:
    results_step_1 = json.load(file)
results_step_2 = dict()
results_step_2["union"] = run_experiment_step_2(results_step_1["union"], hyperparameters["L"], hyperparameters["Z"], hyperparameters["W"], hyperparameters["beta"], generate_random_noise, f, joint_method = "union")
results_step_2["max"] = run_experiment_step_2(results_step_1["max"], hyperparameters["L"], hyperparameters["Z"], hyperparameters["W"], hyperparameters["beta"], generate_random_noise, f, joint_method = "max")
with open("case_studies_results/results_case_study_4/results_step_2.json", "w") as file:
    json.dump(results_step_2, file)

