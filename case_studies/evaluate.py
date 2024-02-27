"""
In this file, we implement the general experimental procedure from the paper detailed in Section 6.
"""

# Import necessary modules.
import sys
from case_studies_configuration import system_path
sys.path.append('resources/')
sys.path.append(system_path)
import numpy as np
from solver import solve
from robust_conformal_prediction import calculate_delta_tilde, phi
from configuration import *

# Hyperparameter setting:
np.random.seed(config_seed)


def run_experiment_step_1(N, K, V, method, delta, training_noise_generator, test_noise_generator, hs, gs, x_dim, f, J, f_value, J_value, omega = None, robust = False, epsilon = None, joint_method = None):
    """
    Run the first step of the experiment.
    :param N: the number of repetitions of the experiment.
    :param K: the number of training data.
    :param V: the number of test data.
    :param method: the method to be used. Choices include "SA", "SAA", "CPP-KKT", and "CPP-MIP".
    :param delta: the expected miscoverage rate.
    :param training_noise_generator: the noise generator for the training data.
    :param test_noise_generator: the noise generator for the test data.
    :param hs: the list of deterministic inequality constraint functions, should be a function of x only and upper bounded by 0.
    :param gs: the list of deterministic equality constraint functions, should be a function of x only and equal to 0.
    :param x_dim: the dimension of the decision variable x.
    :param f: the chance constraint function (compatible wih SCIP), should be a function of x and Y and upper bounded by 0. Alternatively, this can be a list of functions in the case of JCCO (Note this requires that the function constraints satisfy simultaneously).
    :param J: the cost function (compatible wih SCIP), should be a function of x only.
    :param f_value: the chance constraint function (that returns the value), should be a function of x and Y. Alternatively, this can be a list of functions in the case of JCCO (Note this requires that the function constraints satisfy simultaneously).
    :param J_value: the cost function (that returns the value), should be a function of x only.
    :param omega: the omega parameter for SAA.
    :param robust: true or false for robust vs. not robust.
    :param epsilon: the distribution shift to be handled by the robust encoding (in KL divergence).
    :return: the statistics as the results of the experiment.
    """
    # Check for the usage of omega.
    if method == "SAA" and omega is None:
        raise Exception("The omega parameter is not set for SAA.")
    if method == "SAA" and (omega <= 0 or omega >= 1):
        raise Exception("The omega parameter should be in the range (0, 1).")

    # Check that f_value and f are the same type.
    if callable(f) and not callable(f_value):
        raise Exception("The function f and f_value should be the same type.")
    if not callable(f) and callable(f_value):
        raise Exception("The function f and f_value should be the same type.")

    # Check for joint.
    if not callable(f_value):
        # Check that no robust flag is set.
        if robust or (epsilon is not None):
            raise Exception("Robust encoding is not supported for JCCO.")
        if joint_method is None:
            raise Exception("The joint method is not set for JCCO.")

    # Record the statistics.
    num_infeasible = 0
    num_timeout = 0
    solver_times = []
    optimal_solutions = []
    optimal_values = []
    final_test_ys = []
    empirical_coverages = []

    # Compute solutions.
    for n in range(N):
        print("Performing: CPP Step 1 with n = " + str(n + 1))
        # Generate the training data.
        training_Ys = [training_noise_generator() for i in range(K)]
        # Run the optimization.
        x_opt, solver_time = solve(x_dim, delta, training_Ys, hs, gs, f, J, method, omega = omega, robust = robust, epsilon = epsilon, joint_method = joint_method)
        # Handle the error and infeasibility.
        if type(x_opt) == str and x_opt == "infeasible":
            print("Warning: Infeasibility to the Quantile Reformulation detected.")
            num_infeasible += 1
            continue
        elif type(x_opt) == str and x_opt == "timelimit":
            print("Warning: Timeout")
            num_timeout += 1
            continue
        elif type(x_opt) == str:
            raise Exception(f"Error: Error in optimization occured: {x_opt}")
        # Record the solver time.
        solver_times.append(solver_time)
        # Record the optimal solution.
        optimal_solutions.append(x_opt)
        # Record the optimal value.
        optimal_values.append(J_value(x_opt))
        # Check feasibility.
        test_Ys = [test_noise_generator() for i in range(V)]
        final_test_ys.append(test_Ys)
        feasible_count = 0
        if callable(f_value):
            for Y in test_Ys:
                if f_value(x_opt, Y) <= 0:
                    feasible_count += 1
        else:
            for Y in test_Ys:
                if all([f_value[j](x_opt, Y) <= 0 for j in range(len(f_value))]):
                    feasible_count += 1
        empirical_coverages.append(feasible_count / V)

    # Summarize the statistics.
    statistics = dict()
    statistics["solver_times"] = solver_times
    statistics["optimal_solutions"] = optimal_solutions
    statistics["optimal_values"] = optimal_values
    statistics["num_infeasible"] = num_infeasible
    statistics["num_timeout"] = num_timeout
    statistics["empirical_coverages"] = empirical_coverages
    statistics["final_test_Ys"] = final_test_ys
    statistics["N"] = N
    statistics["K"] = K
    statistics["V"] = V
    statistics["delta"] = delta
    statistics["method"] = method
    statistics["omega"] = omega
    return statistics


def run_experiment_step_2(statistics, L, training_noise_generator, f_value, robust = False, epsilon = None, joint_method = None):
    """
    Run the second step of the experiment.
    :param statistics: the statistics from the first step of the experiment.
    :param L: the number of calibration data.
    :param training_noise_generator: the noise generator for the training data.
    :param f_value: the chance constraint function (that returns the value), should be a function of x and Y.  Alternatively, this can be a list of functions in the case of JCCO (Note this requires that the function constraints satisfy simultaneously).
    :param robust: the robustness flag.
    :param epsilon: the distribution shift to be handled by the robust encoding (in KL divergence).
    :return: the statistics as the results of the experiment.
    """
    # Check for joint.
    if not callable(f_value):
        # Check that no robust flag is set.
        if robust or (epsilon is not None):
            raise Exception("Robust encoding is not supported for JCCO.")
        if joint_method is None:
            raise Exception("The joint method is not set for JCCO.")

    step_2_statistics = dict()
    Cs = []
    posterior_coverages = []
    for i in range(len(statistics["optimal_solutions"])):
        # Compute the calibration data.
        x_opt = statistics["optimal_solutions"][i]
        calibration_Ys = [training_noise_generator() for l in range(L)]
        if callable(f_value):
            calibration_fs = [f_value(x_opt, Y) for Y in calibration_Ys]
            calibration_fs.sort()
            if robust:
                delta_tilde = calculate_delta_tilde(statistics["delta"], L, phi, epsilon)
                p = int(np.ceil(L * (1 - delta_tilde)))
            else:
                p = int(np.ceil((L + 1) * (1 - statistics["delta"])))
            c = calibration_fs[p - 1]
        elif joint_method == "Union":
            c_collection = []
            for j in range(len(f_value)):
                calibration_fs_j = [f_value[j](x_opt, Y) for Y in calibration_Ys]
                calibration_fs_j.sort()
                p = int(np.ceil((L + 1) * (1 - statistics["delta"] / len(f_value))))
                c_collection.append(calibration_fs_j[p - 1])
            c = max(c_collection)
        else:
            calibration_fs = [max([f_value[j](x_opt, Y) for j in range(len(f_value))]) for Y in calibration_Ys]
            calibration_fs.sort()
            p = int(np.ceil((L + 1) * (1 - statistics["delta"])))
            c = calibration_fs[p - 1]
        Cs.append(c)
        # Check posterior feasibility.
        feasible_count = 0
        if callable(f_value):
            for Y in statistics["final_test_Ys"][i]:
                if f_value(x_opt, Y) <= Cs[i]:
                    feasible_count += 1
        else:
            for Y in statistics["final_test_Ys"][i]:
                if all([f_value[j](x_opt, Y) <= Cs[i] for j in range(len(f_value))]):
                    feasible_count += 1
        posterior_coverages.append(feasible_count / statistics["V"])

    # Summarize the statistics.
    step_2_statistics["L"] = L
    step_2_statistics["K"] = statistics["K"]
    step_2_statistics["Cs"] = Cs
    step_2_statistics["posterior_coverages"] = posterior_coverages
    return step_2_statistics