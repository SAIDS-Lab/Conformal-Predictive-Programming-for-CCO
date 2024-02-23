"""
In this file, we implement the JCCO problem (Appendix D.5 from the paper).
"""

# Import necessary modules.
import numpy as np
import config
from case_studies.evaluate import run_experiment_step_1, run_experiment_step_2
import scipy.stats as stats
import json


# Experimental setting:
np.random.seed(config.config_seed)
A = np.array([[3, -12, 2], [-10, 3, 5], [5, 3, -15]])
N_cons = 3


def generate_random_vector():
    """
    Generate a random vector.
    :return: a random vector.
    """
    return stats.t.rvs(10, 0, 1, size=3).tolist()

def f0(x, Y):
    return A[0][0]*x[0] + A[0][1]*x[1] + A[0][2]*x[2] - Y[0]


def f1(x, Y):
    return A[1][0]*x[0] + A[1][1]*x[1] + A[1][2]*x[2] - Y[1]


def f2(x, Y):
    return A[2][0]*x[0] + A[2][1]*x[1] + A[2][2]*x[2] - Y[2]


def J(x):
    c = np.array([1, 1, 1])
    J = sum(c[i]*x[i] for i in range(N_cons))
    return J

groups = [{"N": 100, "K": 40, "L": 300, "V": 1000, "delta": 0.2},
            {"N": 100, "K": 80, "L": 300, "V": 1000, "delta": 0.2},
            {"N": 100, "K": 120, "L": 300, "V": 1000, "delta": 0.2}
          ]

f = [f0, f1, f2]
hs = [lambda x: 0 - x[0], lambda x: 0 - x[1], lambda x: 0 - x[2]]
gs = []

# Run the experiment.
results_step_1 = dict()
results_step_1["CPP-MIP Union"] = dict()
results_step_1["CPP-MIP Max"] = dict()
for hyperparameters in groups:
    print("Processing Hyperparameters:", hyperparameters)
    print("Evaluating with CPP-MIP with Union:")
    results_step_1["CPP-MIP Union"][hyperparameters["K"]] = run_experiment_step_1(hyperparameters["N"], hyperparameters["K"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], generate_random_vector, generate_random_vector, hs, gs, 3, f, J, f, J, joint_method = "Union")
    print("Evaluating with CPP-MIP with Max:")
    results_step_1["CPP-MIP Max"][hyperparameters["K"]] = run_experiment_step_1(hyperparameters["N"], hyperparameters["K"], hyperparameters["V"], "CPP-MIP", hyperparameters["delta"], generate_random_vector, generate_random_vector, hs, gs, 3, f, J, f, J, joint_method = "Max")
    print()
with open("../case_study_results/results_case_study_4/results_step_1.json", "w") as file:
    json.dump(results_step_1, file)

# Run the second step of the experiment.
print("Performing the second step of the experiment.")
results_step_2 = dict()
for j_method in ["CPP-MIP Union", "CPP-MIP Max"]:
    results_step_2[j_method] = dict()
    for hyperparameters in groups:
        results_step_2[j_method][hyperparameters["K"]] = run_experiment_step_2(results_step_1[j_method][hyperparameters["K"]], hyperparameters["L"], generate_random_vector, f, joint_method = j_method)
with open("../case_study_results/results_case_study_4/results_step_2.json", "w") as file:
    json.dump(results_step_2, file)