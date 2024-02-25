"""
In this file we implement the functions for robust conformal prediction.
"""

# Import necessary modules.
import sys
sys.path.append('Conformal-Predictive-Programming-for-CCO/')
sys.path.append('resources/')
import math
import numpy as np
import configuration as config

# Hyperparameter setting:
np.random.seed(config.config_seed)


# Functions for robust conformal prediction.
def phi(t):
    """
    The phi function from robust conformal prediction (we use the KL-divergence).

    :param t: the input argument.
    :return: the function value.
    """
    # We assume to use the KL-divergence.
    return t * math.log(t)


def v(phi, epsilon, beta, search_step=0.0007):
    """
    The v function from robust conformal prediction.

    :param phi: the phi function from the robust conformal prediction.
    :param epsilon: the distribution shift to be handled by the robust encoding (in KL divergence).
    :param beta: the input argument.
    :param search_step: the search step for the optimization problem.
    :return: the function value.
    """
    # Check input.
    if beta < 0 or beta > 1:
        raise Exception("Input to the function v is out of range.")

    # Perform a sampling-based line search.
    z = search_step
    while z <= 1:
        value = beta * phi(z / beta) + (1 - beta) * phi((1 - z) / (1 - beta))
        if value <= epsilon:
            return z
        z += search_step

    raise Exception("No return from function v.")


def v_inverse(phi, epsilon, tau, search_step=0.0007):
    """
    The v_inverse function from robust conformal prediction.

    :param phi: the phi function from the robust conformal prediction.
    :param epsilon: the distribution shift to be handled by the robust encoding (in KL divergence).
    :param tau: the input argument.
    :param search_step: the search step for the optimization problem.
    :return: the function value.
    """
    # Check input.
    if tau < 0 or tau > 1:
        raise Exception("Input to the function v_inverse is out of range.")

    beta = 1
    while beta >= 0:
        if beta != 1 and v(phi, epsilon, beta) <= tau:
            return beta
        beta -= search_step

    raise Exception("No return from function v_inverse.")


def calculate_delta_n(delta, L, phi, epsilon):
    """
    The delta_n function from robust conformal prediction.

    :param delta: the expected miscoverage rate.
    :param L: the training or calibration data size.
    :param phi: the phi function from the robust conformal prediction.
    :param epsilon: the distribution shift to be handled by the robust encoding (in KL divergence).
    :return: the function value.
    """
    inner = (1 + 1 / L) * v_inverse(phi, epsilon, 1 - delta)
    return 1 - v(phi, epsilon, inner)


def calculate_delta_tilde(delta, L, phi, epsilon):
    """
    The delta_tilde function from robust conformal prediction.

    :param delta: the expected miscoverage rate.
    :param L: the training or calibration data size.
    :param phi: the phi function from the robust conformal prediction.
    :param epsilon: the distribution shift to be handled by the robust encoding (in KL divergence).
    :return: the function value.
    """
    delta_n = calculate_delta_n(delta, L, phi, epsilon)
    delta_tilde = 1 - v_inverse(phi, epsilon, 1 - delta_n)
    print("the delta_tilde is:", delta_tilde)
    return delta_tilde


def check_data_num(L, delta, epsilon):
    """
    Check if the data size is large enough for robust conformal prediction.

    :param L: the training or calibration data size.
    :param delta: the expected miscoverage rate.
    :param epsilon: the distribution shift to be handled by the robust encoding (in KL divergence).
    """
    data_num_min = np.ceil(v_inverse(phi, epsilon, 1 - delta) / (1 - v_inverse(phi, epsilon, 1 - delta)))
    if L < data_num_min:
        raise Exception(f"Training or Calibration dataset is not large enough. We only have {L} data and we need {data_num_min} data")