"""
In this file, we implement the general experimental procedure from the paper detailed in Section 6.
"""

# Import necessary modules.
import numpy as np
import config
from resources.solvers import cco_solve

# Hyperparameter setting:
np.random.seed(config.config_seed)


def run_experiment_step_1(N, K, V, method, delta, noise_generator, hs, gs, x_dim, f, J, f_value, J_value, omega = None):
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
        training_Ys = [noise_generator() for i in range(K)]
        # Run the optimization.
        x_opt, solver_time = cco_solve(x_dim, delta, training_Ys, hs, gs, f, J, method, omega = omega)
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
        test_Ys = [noise_generator() for i in range(V)]
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