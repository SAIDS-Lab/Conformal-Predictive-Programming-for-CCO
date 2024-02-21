"""
In this file, we implement the general nonlinear CCO problem (case 1 from the paper).
"""

# Import necessary modules.
import numpy as np
import config
from case_studies.evaluate import run_experiment_step_1, run_experiment_step_2
from pyscipopt import exp
import math
import json


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
group_parameters = [{"N": 300, "K": 50, "V": 1000, "delta": 0.05, "saa_omega_1": 0.01, "saa_omega_2":0.03},
                    {"N": 300, "K": 100, "V": 1000, "delta": 0.05, "saa_omega_1": 0.01, "saa_omega_2":0.03},
                    {"N": 300, "K": 200, "V": 1000, "delta": 0.05, "saa_omega_1": 0.01, "saa_omega_2": 0.03},
                    {"N": 300, "K": 300, "V": 1000, "delta": 0.05, "saa_omega_1": 0.01, "saa_omega_2": 0.03},
                    {"N": 300, "K": 500, "V": 1000, "delta": 0.05, "saa_omega_1": 0.01, "saa_omega_2": 0.03}]
num_groups = len(group_parameters)
Ls = [50, 200, 500, 750, 1000]

# Run the first step of the experiment.
results_step_1 = dict()
print("Performing the first step of the experiment with the specified group parameters.")
for i in range(num_groups):
    parameters = group_parameters[i]
    print("Evaluating with parameters:", parameters)
    results_step_1[i] = dict()
    print("Evaluating with CPP-KKT:")
    results_step_1[i]["CPP-KKT"] = run_experiment_step_1(parameters["N"], parameters["K"], parameters["V"], "CPP-KKT", parameters["delta"], generate_random_noise, generate_random_noise, hs, gs, 1, f, J, f_value, J_value)
    print("Evaluating with CPP-MIP:")
    results_step_1[i]["CPP-MIP"] = run_experiment_step_1(parameters["N"], parameters["K"], parameters["V"], "CPP-MIP", parameters["delta"], generate_random_noise, generate_random_noise, hs, gs, 1, f, J, f_value, J_value)
    print("Evaluating with SA:")
    results_step_1[i]["SA"] = run_experiment_step_1(parameters["N"], parameters["K"], parameters["V"], "SA", parameters["delta"], generate_random_noise, generate_random_noise, hs, gs, 1, f, J, f_value, J_value)
    print("Evaluating with SAA with omega 1:" + str(parameters["saa_omega_1"]))
    results_step_1[i]["SAA_1"] = run_experiment_step_1(parameters["N"], parameters["K"], parameters["V"], "SAA", parameters["delta"], generate_random_noise, generate_random_noise, hs, gs, 1, f, J, f_value, J_value, parameters["saa_omega_1"])
    print("Evaluating with SAA with omega 2:" + str(parameters["saa_omega_2"]))
    results_step_1[i]["SAA_2"] = run_experiment_step_1(parameters["N"], parameters["K"], parameters["V"], "SAA", parameters["delta"], generate_random_noise, generate_random_noise, hs, gs, 1, f, J, f_value, J_value, parameters["saa_omega_2"])
    print()

# Save the results from the first step.
print("Saving the results from the first step.")
for i in range(num_groups):
    with open("../case_study_results/results_case_study_1/results_step_1_case_1_K=" + str(group_parameters[i]["K"]) + ".json", "w") as file:
        json.dump(results_step_1[i], file)
print()

# Run the second step of the experiment.
print("Performing the second step of the experiment with the specified calibration parameters.")
results_step_2 = dict()
for i in range(num_groups):
    current_k = group_parameters[i]["K"]
    results_step_2[current_k] = dict()
    for L in Ls:
        results_step_2[current_k][L] = dict()
        for method in ["CPP-KKT", "CPP-MIP", "SA", "SAA_1", "SAA_2"]:
            results_step_2[current_k][L][method] = run_experiment_step_2(results_step_1[i][method], L, generate_random_noise, f_value)
with open("../case_study_results/results_case_study_1/results_step_2_case_1.json", "w") as file:
    json.dump(results_step_2, file)