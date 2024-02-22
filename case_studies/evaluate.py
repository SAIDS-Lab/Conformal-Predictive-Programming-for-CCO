"""
In this file, we implement the general experimental procedure from the paper detailed in Section 6.
"""

# Import necessary modules.
import numpy as np
import config
from resources.solvers import cco_solve
from resources.robust_conformal_prediction import calculate_delta_tilde, phi

# Hyperparameter setting:
np.random.seed(config.config_seed)


def run_experiment_step_1(N, K, V, method, delta, training_noise_generator, test_noise_generator, hs, gs, x_dim, f, J, f_value, J_value, omega = None, robust = False, epsilon = None):
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
    :param f: the chance constraint function (compatible wih SCIP), should be a function of x and Y and upper bounded by 0.
    :param J: the cost function (compatible wih SCIP), should be a function of x only.
    :param f_value: the chance constraint function (that returns the value), should be a function of x and Y.
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
        x_opt, solver_time = cco_solve(x_dim, delta, training_Ys, hs, gs, f, J, method, omega = omega, robust = robust, epsilon = epsilon)
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
        for Y in test_Ys:
            if f_value(x_opt, Y) <= 0:
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


def run_experiment_step_2(statistics, L, training_noise_generator, f_value, robust = False, epsilon = None):
    """
    Run the second step of the experiment.
    :param statistics: the statistics from the first step of the experiment.
    :param L: the number of calibration data.
    :param training_noise_generator: the noise generator for the training data.
    :param f_value: the chance constraint function (that returns the value), should be a function of x and Y.
    :param robust: the robustness flag.
    :param epsilon: the distribution shift to be handled by the robust encoding (in KL divergence).
    :return: the statistics as the results of the experiment.
    """
    step_2_statistics = dict()
    Cs = []
    posterior_coverages = []
    for i in range(len(statistics["optimal_solutions"])):
        # Compute the calibration data.
        x_opt = statistics["optimal_solutions"][i]
        calibration_Ys = [training_noise_generator() for l in range(L)]
        calibration_fs = [f_value(x_opt, Y) for Y in calibration_Ys]
        calibration_fs.sort()
        if robust:
            delta_tilde = calculate_delta_tilde(statistics["delta"], L, phi, epsilon)
            p = int(np.ceil(L * (1 - delta_tilde)))
        else:
            p = int(np.ceil((L + 1) * (1 - statistics["delta"])))
        c = calibration_fs[p - 1]
        Cs.append(c)
        # Check posterior feasibility.
        feasible_count = 0
        for Y in statistics["final_test_Ys"][i]:
            if f_value(x_opt, Y) <= Cs[i]:
                feasible_count += 1
        posterior_coverages.append(feasible_count / statistics["V"])

    # Summarize the statistics.
    step_2_statistics["L"] = L
    step_2_statistics["K"] = statistics["K"]
    step_2_statistics["Cs"] = Cs
    step_2_statistics["posterior_coverages"] = posterior_coverages
    return step_2_statistics