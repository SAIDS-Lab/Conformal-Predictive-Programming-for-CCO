"""
In this file, we implement the optimal control problem (case 2 from the paper).
"""

# Import necessary modules.
import numpy as np
from evaluate import run_experiment_step_1, run_experiment_step_2
import json
import configuration as config
import random
import math

# Experimental setting:
# np.random.seed(config.config_seed)
np.random.seed(12)
T = 5
z = (5, 5)
zeta = 1
hyperparameters = {"N": 300, "K": 200, "L": 200, "V": 1000, "Z": 300, "W": 1000, "delta": 0.1, "beta": 0.1}


def generate_random_noise_matrix():
    """
    Generate a random noise matrix.
    :return: a random noise matrix.
    """
    # return random.uniform(15, 25)    # 0.013 17.1  19.8  (300)
    # return np.random.laplace(50, 1)    # 0.001 0.2 0.13 (30)

    # return random.uniform(0, 1)    # 0.008 5.5 3.9 (30)
    return random.uniform(15, 16)    # 0.013 30 21 (30)
    # return random.uniform(15, 20)   # 0.02 27 43 (30)     0.02 31  46 (30)
    # return random.uniform(15, 30)  # 0.017 22.73 12.99 (30)

    # return np.random.laplace(20, 1)   # 0.29 11 19  (30)
    # return np.random.laplace(20, 2.5)   # 0.28 10 11  (30)

    # return np.random.laplace(50, 1)    # 0.001 0.2 0.13 (30)
    # return np.random.laplace(50, 2.5)    # 0.005 0.25 0.14 (30)
    # return np.random.laplace(50, 5)    # 0.04 0.44 0.17 (30)
    
    
    

def f(x, Y):
    """
    The function f from the optimal control problem.

    :param x: the decision variable
    :param Y: the noise.
    :return: the function value.
    """
    return (x[0] - 3)**2 + (x[1] - 5)**2 - Y

def f_value(x, Y):
    """
    The function f from the optimal control problem.

    :param x: the decision variable
    :param Y: the noise.
    :return: the function value.
    """
    return (x[0] - 3)**2 + (x[1] - 5)**2 - Y

gs = []
hs = []


def J(x):
    """
    The cost function J from the optimal control problem.

    :param x: the decision variable.
    :return: the function value.
    """
    return - x[0] - 2*x[1]


def J_value(x):
    """
    The cost function J from the optimal control problem.

    :param x: the decision variable.
    :return: the function value.
    """
    return - x[0] - 2*x[1]


# Run the first step of the experiment.
results_step_1 = dict()
print("Evaluating with CPP-Discard:")
results_step_1["CPP-Discard_m"] = run_experiment_step_1("marginal", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-Discard", hyperparameters["delta"], hyperparameters["beta"], generate_random_noise_matrix, generate_random_noise_matrix, hs, gs, 2, f, J, f_value, J_value)
results_step_1["CPP-Discard_c"] = run_experiment_step_1("conditional", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-Discard", hyperparameters["delta"], hyperparameters["beta"], generate_random_noise_matrix, generate_random_noise_matrix, hs, gs, 2, f, J, f_value, J_value)
print()

print("Evaluating with CPP-KKT:")
results_step_1["CPP-KKT_m"] = run_experiment_step_1("marginal", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-KKT", hyperparameters["delta"], hyperparameters["beta"], generate_random_noise_matrix, generate_random_noise_matrix, hs, gs, 2, f, J, f_value, J_value)
results_step_1["CPP-KKT_c"] = run_experiment_step_1("conditional", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-KKT", hyperparameters["delta"], hyperparameters["beta"], generate_random_noise_matrix, generate_random_noise_matrix, hs, gs, 2, f, J, f_value, J_value)
print()

print("Evaluating with CPP-MIP:")
results_step_1["CPP-MIP_m"] = run_experiment_step_1("marginal", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], hyperparameters["beta"], generate_random_noise_matrix, generate_random_noise_matrix, hs, gs, 2, f, J, f_value, J_value)
results_step_1["CPP-MIP_c"] = run_experiment_step_1("conditional", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], hyperparameters["beta"], generate_random_noise_matrix, generate_random_noise_matrix, hs, gs, 2, f, J, f_value, J_value)
print()


# Save the results for the first step of the experiment.
with open("case_studies_results/results_case_study_1/results_step_1.json", "w") as file:
    json.dump(results_step_1, file)

# with open("case_studies_results/results_case_study_1/time_test.json", "w") as file:
#     json.dump(results_step_1, file)
# print("K = ", hyperparameters["K"])
# print("Discard average time:", sum(results_step_1["CPP-Discard_c"]["solver_times"]) / len(results_step_1["CPP-Discard_c"]["solver_times"]))
# print("KKT average time:", sum(results_step_1["CPP-KKT_c"]["solver_times"]) / len(results_step_1["CPP-KKT_c"]["solver_times"]))
# print("MIP average time:", sum(results_step_1["CPP-MIP_c"]["solver_times"]) / len(results_step_1["CPP-MIP_c"]["solver_times"]))


# Run the second step of the experiment.
with open("case_studies_results/results_case_study_1/results_step_1.json", "r") as file:
    results_step_1 = json.load(file)
results_step_2 = dict()
results_step_2["CPP-Discard"] = run_experiment_step_2(results_step_1["CPP-Discard_m"], results_step_1["CPP-Discard_c"], hyperparameters["L"], hyperparameters["Z"], hyperparameters["W"], hyperparameters["beta"], generate_random_noise_matrix, f_value)
results_step_2["CPP-KKT"] = run_experiment_step_2(results_step_1["CPP-KKT_m"], results_step_1["CPP-KKT_c"], hyperparameters["L"], hyperparameters["Z"], hyperparameters["W"], hyperparameters["beta"], generate_random_noise_matrix, f_value)
results_step_2["CPP-MIP"] = run_experiment_step_2(results_step_1["CPP-MIP_m"], results_step_1["CPP-MIP_c"], hyperparameters["L"], hyperparameters["Z"], hyperparameters["W"], hyperparameters["beta"], generate_random_noise_matrix, f_value)
# Save the results for the second step of the experiment.
with open("case_studies_results/results_case_study_1/results_step_2.json", "w") as file:
    json.dump(results_step_2, file)