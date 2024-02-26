"""
In this file, we implement the portfolio optimization problem (case 3 from the paper).
"""

# Import necessary modules.
import numpy as np
from evaluate import run_experiment_step_1, run_experiment_step_2
import json
import configuration as config


# Experimental setting:
np.random.seed(config.config_seed)
x_num = 3             # the number of the assets
w_total = 100    # the total assets
w_min = 10
eta = 350      # risk tolerance
T = np.array([[0.06, 0.07, 0.02], [0.09, 0.05, 0.03], [0.03, 0.02, 0.01]]) # risk weight matrix


mean1 = [0.12, 0.1, 0.07]
var1 = [[0.013, 0, 0], [0, 0.01, 0], [0, 0, 0.008]]
mean2 = [0.1, 0.11, 0.07]
var2 = [[0.013, 0, 0], [0, 0.011, 0], [0, 0, 0.007]]


def generate_random_noise_matrix_nominal():
    """
    Generate a random noise matrix for the nominal distribution.
    :return: a random noise matrix.
    """
    return np.random.multivariate_normal(mean1, var1).tolist()


def generate_random_noise_matrix_test():
    """
    Generate a random noise matrix for the test distribution.
    :return: a random noise matrix.
    """
    return np.random.multivariate_normal(mean2, var2).tolist()


def f(x, Y):
    """
    Let x[3] represent theta.
    """
    return x[3] - x[0] * Y[0] - x[1] * Y[1] - x[2] * Y[2]


def J(x):
    return 0 - x[3]


def h_1(x):
    nonlinear_expr = 0
    for i in range(x_num):
        for j in range(x_num):
            nonlinear_expr += T[i][j] * x[i] * x[j]
    return nonlinear_expr - eta


def h_2(x):
    return sum(x[i] for i in range(x_num)) - w_total


def h_3(x):
    return 0 - x[3]


def h_4(x):
    return w_min - x[0]


def h_5(x):
    return w_min - x[1]


def h_6(x):
    return w_min - x[2]


def compute_divergence(mean_test, var_test, mean_train, var_train):
    """
    Find the KL-divergence between two Gaussian distributions.

    :param mean_test: the mean of the test distribution.
    :param var_test: the covariance of the test distribution.
    :param mean_train: the mean of the training distribution.
    :param var_train: the covariance of the training distribution.
    :return: the KL divergence.
    """
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    inv_cov_test = np.linalg.inv(var_test)
    trace_term = np.trace(np.dot(inv_cov_test, var_train)-I)
    diff_mean = mean_test - mean_train
    quad_form = np.dot(diff_mean.T, np.dot(inv_cov_test, diff_mean))
    log_det_ratio = np.log(np.linalg.det(np.dot(var_train, inv_cov_test)))
    kl_divergence = 0.5 * (trace_term + quad_form - log_det_ratio)
    print("the KL-divergence is:", kl_divergence)
    return kl_divergence


hs = [h_1, h_2, h_3, h_4, h_5, h_6]
gs = []
hyperparameters = {"N": 100, "K": 75, "L": 200, "V": 1000, "delta": 0.2}
epsilon = compute_divergence(np.array(mean2), np.array(var2), np.array(mean1), np.array(var1))


# Run the experiment.
results_step_1 = dict()
print("Evaluating with CPP-KKT (Robust):")
results_step_1["CPP-KKT Robust"] = run_experiment_step_1(hyperparameters["N"], hyperparameters["K"], hyperparameters["V"], "CPP-KKT", hyperparameters["delta"], generate_random_noise_matrix_nominal, generate_random_noise_matrix_test, hs, gs, 4, f, J, f, J, robust = True, epsilon = epsilon)
print()

print("Evaluating with CPP-MIP (Robust):")
results_step_1["CPP-MIP Robust"] = run_experiment_step_1(hyperparameters["N"], hyperparameters["K"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], generate_random_noise_matrix_nominal, generate_random_noise_matrix_test, hs, gs, 4, f, J, f, J, robust = True, epsilon = epsilon)
print()

print("Evaluating with CPP-KKT:")
results_step_1["CPP-KKT"] = run_experiment_step_1(hyperparameters["N"], hyperparameters["K"], hyperparameters["V"], "CPP-KKT", hyperparameters["delta"], generate_random_noise_matrix_nominal, generate_random_noise_matrix_test, hs, gs, 4, f, J, f, J)
print()

print("Evaluating with CPP-MIP:")
results_step_1["CPP-MIP"] = run_experiment_step_1(hyperparameters["N"], hyperparameters["K"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], generate_random_noise_matrix_nominal, generate_random_noise_matrix_test, hs, gs, 4, f, J, f, J)
print()

with open("case_studies_results/results_case_study_3/results_step_1.json", "w") as file:
    json.dump(results_step_1, file)

# Run the second step of the experiment.
results_step_2 = dict()
results_step_2["CPP-KKT Robust"] = run_experiment_step_2(results_step_1["CPP-KKT"], hyperparameters["L"], generate_random_noise_matrix_nominal, f, robust = True, epsilon = epsilon)
results_step_2["CPP-MIP Robust"] = run_experiment_step_2(results_step_1["CPP-MIP"], hyperparameters["L"], generate_random_noise_matrix_nominal, f, robust = True, epsilon = epsilon)
results_step_2["CPP-KKT"] = run_experiment_step_2(results_step_1["CPP-KKT"], hyperparameters["L"], generate_random_noise_matrix_nominal, f)
results_step_2["CPP-MIP"] = run_experiment_step_2(results_step_1["CPP-MIP"], hyperparameters["L"], generate_random_noise_matrix_nominal, f)

with open("case_studies_results/results_case_study_3/results_step_2.json", "w") as file:
    json.dump(results_step_2, file)