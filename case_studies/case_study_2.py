"""
In this file, we implement the optimal control problem (case 2 from the paper).
"""

# Import necessary modules.
import numpy as np
from evaluate import run_experiment_step_1, run_experiment_step_2
import json
import configuration as config

# Experimental setting:
np.random.seed(config.config_seed)
T = 5
z = (5, 5)
zeta = 1
hyperparameters = {"N": 100, "K": 70, "L": 200, "V": 1000, "delta": 0.1}


def generate_random_noise_matrix():
    """
    Generate a random noise matrix.
    :return: a random noise matrix.
    """
    return np.random.laplace(0, 0.02, (T, 4)).tolist()


def f(u, W):
    """
    The function f from the optimal control problem.

    :param u: the decision variable
    :param W: the noise.
    :return: the function value.
    """
    y0 = np.array([0, 0, 0, 0])
    A = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    B = np.array([[0.5, 0], [1, 0], [0, 0.5], [0, 1]])
    ys = [y0]
    for t in range(T):
        y_new = A @ ys[-1] + B @ np.array([u[t, 0], u[t, 1]]) + W[t]
        ys.append(y_new)
    yT = ys[-1]
    return (yT[0] - z[0]) * (yT[0] - z[0]) + (yT[2] - z[1]) * (yT[2] - z[1]) - zeta
gs = []
hs = []


def J(u):
    """
    The cost function J from the optimal control problem.

    :param u: the decision variable.
    :return: the function value.
    """
    return sum(u[t, 0] * u[t, 0] + u[t, 1] * u[t, 1] for t in range(T))


def J_value(u):
    """
    The cost function J from the optimal control problem.

    :param u: the decision variable.
    :return: the function value.
    """
    return sum(u[t][0] * u[t][0] + u[t][1] * u[t][1] for t in range(T))


def f_value(u, W):
    """
    The function f from the optimal control problem.

    :param u: the decision variable
    :param W: the noise.
    :return: the function value.
    """
    y0 = np.array([0, 0, 0, 0])
    A = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    B = np.array([[0.5, 0], [1, 0], [0, 0.5], [0, 1]])
    ys = [y0]
    for t in range(T):
        y_new = A @ ys[-1] + B @ np.array([u[t][0], u[t][1]]) + W[t]
        ys.append(y_new)
    yT = ys[-1]
    return (yT[0] - z[0]) * (yT[0] - z[0]) + (yT[2] - z[1]) * (yT[2] - z[1]) - zeta


# Run the experiment.
results_step_1 = dict()
print("Evaluating with CPP-KKT:")
results_step_1["CPP-KKT"] = run_experiment_step_1(hyperparameters["N"], hyperparameters["K"], hyperparameters["V"], "CPP-KKT", hyperparameters["delta"], generate_random_noise_matrix, generate_random_noise_matrix, hs, gs, (T, 2), f, J, f_value, J_value)
print()

print("Evaluating with CPP-MIP:")
results_step_1["CPP-MIP"] = run_experiment_step_1(hyperparameters["N"], hyperparameters["K"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], generate_random_noise_matrix, generate_random_noise_matrix, hs, gs, (T, 2), f, J, f_value, J_value)
# Save the results for the first step of the experiment.
with open("case_studies_results/results_case_study_2/results_step_1.json", "w") as file:
    json.dump(results_step_1, file)
print()

# Run the second step of the experiment.
results_step_2 = dict()
results_step_2["CPP-KKT"] = run_experiment_step_2(results_step_1["CPP-KKT"], hyperparameters["L"], generate_random_noise_matrix, f_value)
results_step_2["CPP-MIP"] = run_experiment_step_2(results_step_1["CPP-MIP"], hyperparameters["L"], generate_random_noise_matrix, f_value)
# Save the results for the second step of the experiment.
with open("case_studies_results/results_case_study_2/results_step_2.json", "w") as file:
    json.dump(results_step_2, file)