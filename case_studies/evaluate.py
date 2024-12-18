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
import math
from pyscipopt import Model
import configuration as config
import time


# Hyperparameter setting:
np.random.seed(config_seed)


def run_experiment_step_1(mode, N, K, L, V, method, delta, beta, training_noise_generator, test_noise_generator, hs, gs, x_dim, f, J, f_value, J_value, omega = None, robust = False, epsilon = None, joint_method = None):
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
    statistics = dict()
    statistics["N"] = N
    statistics["K"] = K
    statistics["L"] = L
    statistics["V"] = V
    statistics["delta"] = delta
    statistics["method"] = method
    statistics["omega"] = omega

    # adjust the delta for the conditional case
    if mode == "conditional":
        delta = delta - math.sqrt(math.log(1 / beta) / (2 * K))

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
    final_train_ys = []
    final_test_ys = []
    final_calib_ys = []
    empirical_coverages = []

    # Compute solutions.
    for n in range(N):
        print("Performing: CPP Step 1 with n = " + str(n + 1))
        # Generate the training data.
        training_Ys = [training_noise_generator() for i in range(K)]
        final_train_ys.append(training_Ys)
        # Run the optimization.

        if method != "CPP-Discard":
            x_opt, solver_time = solve(x_dim, delta, training_Ys, hs, gs, f, J, method, omega = omega, robust = robust, epsilon = epsilon, joint_method = joint_method)
        else:
            x_opt, solver_time_once = solve(x_dim, delta, training_Ys, hs, gs, f, J, "CPP-Discard", omega = omega, robust = robust, epsilon = epsilon, joint_method = joint_method)
            solver_time = solver_time_once
            while len(training_Ys) > int(np.ceil((K + 1) * (1 - delta))):
                if x_opt == "infeasible":
                    break
                flag = 0
                for data in training_Ys:
                    if f(x_opt, data) >= -0.00001 and f(x_opt, data) <= 0.00001:   # to avoid numerical issues
                        flag = 1
                        training_Ys.remove(data)
                        break
                if flag == 0:
                    break
                x_opt, solver_time_once = solve(x_dim, delta, training_Ys, hs, gs, f, J, "CPP-Discard", omega = omega, robust = robust, epsilon = epsilon, joint_method = joint_method)
                solver_time += solver_time_once


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
        calib_Ys = [test_noise_generator() for i in range(L)]
        final_calib_ys.append(calib_Ys)
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
    statistics["solver_times"] = solver_times
    statistics["optimal_solutions"] = optimal_solutions
    statistics["optimal_values"] = optimal_values
    statistics["num_infeasible"] = num_infeasible
    statistics["num_timeout"] = num_timeout
    statistics["empirical_coverages"] = empirical_coverages
    statistics["final_train_Ys"] = final_train_ys
    statistics["final_test_Ys"] = final_test_ys
    statistics["final_calib_Ys"] = final_calib_ys
    
    return statistics


def run_experiment_step_2(statistics_m, statistics_c, L, Z, W, beta, noise_generator, f_value, robust = False, epsilon = None, joint_method = None):
    """
    Run the second step of the experiment.
    :param statistics: the statistics from the first step of the experiment.
    :param L: the number of calibration data.
    :param noise_generator: the noise generator for the training data.
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
            

    ################# marginal evaluation #################
    step_2_statistics = dict()
    Cs_m = []
    EC_count = 0
    for i in range(len(statistics_m["optimal_solutions"])):
        # Compute the calibration data.
        x_opt = statistics_m["optimal_solutions"][i]
        calibration_Ys = statistics_m["final_calib_Ys"][i]
        if callable(f_value):
            calibration_fs = [f_value(x_opt, Y) for Y in calibration_Ys]
            calibration_fs.sort()
            if robust:
                delta_tilde = calculate_delta_tilde(statistics_m["delta"], L, phi, epsilon)
                p_m = int(np.ceil(L * (1 - delta_tilde)))
            else:
                p_m = int(np.ceil((L + 1) * (1 - statistics_m["delta"])))
            c_m = calibration_fs[p_m - 1]
        elif joint_method == "Union":
            c_collection = []
            for j in range(len(f_value)):
                calibration_fs_j = [f_value[j](x_opt, Y) for Y in calibration_Ys]
                calibration_fs_j.sort()
                p_m = int(np.ceil((L + 1) * (1 - statistics_m["delta"] / len(f_value))))
                c_collection.append(calibration_fs_j[p_m - 1])
            c_m = max(c_collection)
        else:
            calibration_fs = [max([f_value[j](x_opt, Y) for j in range(len(f_value))]) for Y in calibration_Ys]
            calibration_fs.sort()
            p_m = int(np.ceil((L + 1) * (1 - statistics_m["delta"])))
            c_m = calibration_fs[p_m - 1]
        Cs_m.append(c_m)
        # Check posterior feasibility.
        # EC
        # To Nick: Please add the uncallable case here.
        if callable(f_value):
            Y = statistics_m["final_test_Ys"][i][0] # pick the first data as the test data in the EC
            if f_value(x_opt, Y) <= Cs_m[i]:
                EC_count += 1

    EC = EC_count / (len(statistics_m["optimal_solutions"]))

    ################# conditional evaluation #################
    Cs_c = []
    CEC_c = []
    delta_star = []
    delta_comp_time = []
    for i in range(len(statistics_c["optimal_solutions"])):
        # Compute the calibration data.
        
        x_opt = statistics_c["optimal_solutions"][i]
        calibration_Ys = statistics_c["final_calib_Ys"][i]
        calibration_fs = [f_value(x_opt, Y) for Y in calibration_Ys]
        calibration_fs.sort()
        # To Nick: Please add the p_c and c_c for other cases in this for loop, as the above for loop.
        p_c = int(np.ceil((L + 1) * (1 - statistics_c["delta"] + math.sqrt(math.log(1 / beta) / (2 * L))))) 
        c_c = calibration_fs[p_c - 1]
        Cs_c.append(c_c)
        

        # CEC_{c,i}
        feasible_count = sum(1 for Y in statistics_c["final_test_Ys"][i] if f_value(x_opt, Y) <= Cs_c[i])
        CEC_c.append(feasible_count / statistics_c["V"])

        # compute delta for Theorem 3.5
        time_start = time.time()
        S = sum(1 for Y in calibration_Ys if f_value(x_opt, Y) <= 0)
        delta_star.append(1 - S / (L+1) + math.sqrt(math.log(1 / beta) / (2 * L)))
        time_end = time.time()
        delta_comp_time.append(time_end - time_start)
        if i == 1:
            CEC_0_l_z = []
            for i in range(Z):
                test_Ys = [noise_generator() for _ in range(W)]
                feasible_count = sum(1 for Y in test_Ys if f_value(x_opt, Y) <= 0)
                CEC_0_l_z.append(feasible_count / W)


    # Summarize the statistics.
    step_2_statistics["L"] = L
    step_2_statistics["K"] = statistics_m["K"]
    step_2_statistics["Cs_m"] = Cs_m
    step_2_statistics["Cs_c"] = Cs_c
    step_2_statistics["EC"] = EC
    step_2_statistics["CEC_c"] = CEC_c
    step_2_statistics["CEC_0_l_z"] = CEC_0_l_z
    step_2_statistics["delta_star"] = delta_star
    step_2_statistics["delta_comp_time"] = delta_comp_time
    return step_2_statistics



def compute_delta(statistics, training_ys, hs, gs, x_dim, f, J, beta, K):
    sup = []
    sup_comp_time = []
    delta1_comp_time = []
    delta2_comp_time = []
    delta_star_sa_1 = []
    delta_star_sa_2 = []
    for i in range(len(statistics["optimal_solutions"])):
        print("Computing support with n = " + str(i + 1))
        x_opt = statistics["optimal_solutions"][i]

        # Compute the support.
        time_start = time.time()
        indices_to_remove = []
        for j in range(len(training_ys[i])):
            training_ys_prime = [item for idx, item in enumerate(training_ys[i]) if (idx not in indices_to_remove) and (idx != j)]
            delta = 0.1 # this is a useless parameter in the following function
            x_opt_new, _ = solve(x_dim, delta, training_ys_prime, hs, gs, f, J, "SA", omega = None, robust = False, epsilon = None, joint_method = None)
            if x_opt == x_opt_new: #I do not consider the different cases of x_dim here.
                indices_to_remove.append(j)
        sup.append([item for idx, item in enumerate(training_ys[i]) if idx not in indices_to_remove])
        s_K_star = len(sup[i])
        time_end = time.time()
        sup_comp_time.append(time_end - time_start)

        # Compute delta_1.
        time_start = time.time()
        if s_K_star == K:
            delta_star_sa_1.append(1)
        else:
            delta_star_sa_1.append(1 - (beta/(K*math.comb(K, s_K_star)))**(1 / (K - s_K_star)))
        time_end = time.time()
        delta1_comp_time.append(time_end - time_start)

        # Compute delta_2.
        time_start = time.time()
        coefficients = []
        for m in range(s_K_star, K): 
            coefficients.append((beta / K) * math.comb(m, s_K_star))
        coefficients.append(-math.comb(K, s_K_star))
        polynomial = [0] * (K - s_K_star - len(coefficients)) + coefficients[::-1]
        roots = np.roots(polynomial)
        real_roots = roots[np.isclose(roots.imag, 0)].real 
        real_roots_in_interval = real_roots[(real_roots > 0) & (real_roots < 1)]  # as described in the paper, they would have and only have one real root in (0,1)
        if s_K_star == K:
            delta_star_sa_2.append(1)
        else:
            delta_star_sa_2.append(1 - real_roots_in_interval[0])
        time_end = time.time()
        delta2_comp_time.append(time_end - time_start)

    
    statistics["sup"] = sup
    statistics["sup_comp_time"] = sup_comp_time
    statistics["delta1_comp_time"] = delta1_comp_time
    statistics["delta2_comp_time"] = delta2_comp_time
    statistics["delta_star_1"] = delta_star_sa_1
    statistics["delta_star_2"] = delta_star_sa_2

    return statistics
