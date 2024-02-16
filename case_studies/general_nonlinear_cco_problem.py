"""
In this file, we implement the general nonlinear CCO problem (case 1 from the paper).
"""

# Import necessary modules.
import numpy as np
import config
from evaluate import run_experiment_step_1
from pyscipopt import exp
import math

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

run_experiment_step_1(10, 50, 1000, "SA", 0.05, generate_random_noise, hs, gs, 1, f, J, f_value, J_value)