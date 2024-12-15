"""
In this file, we implement the general nonlinear CCO problem (case 1 from the paper).
"""

# Import necessary modules.
import numpy as np
# import config as config
from evaluate import run_experiment_step_1, run_experiment_step_2
from pyscipopt import exp
import math
import json
import configuration as config


def generate_random_noise():
    """
    Generate a random noise.
    :return: a random noise.
    """
    return np.random.exponential(3)


# Experimental setting:
np.random.seed(config.config_seed)
hs = [lambda x: x ** 3 + 20]
gs = []
f = lambda x, Y: exp(x) * (50 * Y) - 5
f_value = lambda x, Y: math.exp(x) * (50 * Y) - 5
J = lambda x: (x ** 3) * exp(x)
J_value = lambda x: (x ** 3) * math.exp(x)
hyperparameters = {"N": 300, "K": 200, "L": 200, "V": 1000, "Z": 300, "W": 1000, "delta": 0.1, "beta": 0.1}


# Run the first step of the experiment.
results_step_1 = dict()
print("Performing the first step of the experiment.")

results_step_1 = dict()
print("Evaluating with CPP-KKT:")
results_step_1["CPP-KKT_m"] = run_experiment_step_1("marginal", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-KKT", hyperparameters["delta"], hyperparameters["beta"], generate_random_noise, generate_random_noise, hs, gs, 1, f, J, f_value, J_value)
results_step_1["CPP-KKT_c"] = run_experiment_step_1("conditional", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-KKT", hyperparameters["delta"], hyperparameters["beta"], generate_random_noise, generate_random_noise, hs, gs, 1, f, J, f_value, J_value)

print("Evaluating with CPP-MIP:")
results_step_1["CPP-MIP_m"] = run_experiment_step_1("marginal", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], hyperparameters["beta"], generate_random_noise, generate_random_noise, hs, gs, 1, f, J, f_value, J_value)
results_step_1["CPP-MIP_c"] = run_experiment_step_1("conditional", hyperparameters["N"], hyperparameters["K"], hyperparameters["L"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], hyperparameters["beta"], generate_random_noise, generate_random_noise, hs, gs, 1, f, J, f_value, J_value)
print()

# Save the results from the first step.
print("Saving the results from the first step.")
with open("case_studies_results/results_case_study_2/results_step_1.json", "w") as file:
    json.dump(results_step_1, file)
print()

print("KKT average time:", sum(results_step_1["CPP-KKT_c"]["solver_times"]) / len(results_step_1["CPP-KKT_c"]["solver_times"]))
print("MIP average time:", sum(results_step_1["CPP-MIP_c"]["solver_times"]) / len(results_step_1["CPP-MIP_c"]["solver_times"]))


# Run the second step of the experiment.
with open("case_studies_results/results_case_study_2/results_step_1.json", "r") as file:
    results_step_1 = json.load(file)
print("Performing the second step of the experiment with the specified calibration parameters.")
results_step_2 = dict()
results_step_2["CPP-KKT"] = run_experiment_step_2(results_step_1["CPP-KKT_m"], results_step_1["CPP-KKT_c"], hyperparameters["L"], hyperparameters["Z"], hyperparameters["W"], hyperparameters["beta"], generate_random_noise, f_value)
results_step_2["CPP-MIP"] = run_experiment_step_2(results_step_1["CPP-MIP_m"], results_step_1["CPP-KKT_c"], hyperparameters["L"], hyperparameters["Z"], hyperparameters["W"], hyperparameters["beta"], generate_random_noise, f_value)

# Save the results from the second step.
with open("case_studies_results/results_case_study_2/results_step_2.json", "w") as file:
    json.dump(results_step_2, file)


