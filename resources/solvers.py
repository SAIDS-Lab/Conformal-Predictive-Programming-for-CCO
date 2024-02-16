"""
In this file, we write the solver for the CCO/RCCO/JCCO problems, which calls the encoders in quantile_encoders.
"""

# Import necessary modules.
import numpy as np
import time
import config
from resources.quantile_encoders.chance_constraint_encoder import *
from pyscipopt import Model

# Hyperparameter setting:
np.random.seed(config.config_seed)


def cco_solve(x_dim, delta, training_Ys, hs, gs, f, J, method, omega = None):
    """
    Solve the CCO problem.

    :param x_dim: the dimension of the decision variable x.
    :param delta: the expected miscoverage rate.
    :param training_Ys: the training data Y^{(1)}, ..., Y^{(K)}.
    :param hs: the list of deterministic inequality constraint functions, should be a function of x only and upper bounded by 0.
    :param gs: the list of deterministic equality constraint functions, should be a function of x only and equal to 0.
    :param f: the chance constraint function, should be a function of x and Y and upper bounded by 0.
    :param J: the cost function, should be a function of x only.
    :param method: the method used for solving the cco. The acceptable method includes "SA", "SAA", "CPP-KKT", and "CPP-MIP".
    :param omega: the omega parameter for SAA.
    :return: the solution and the time used for solving the problem.
    """
    # Initialize the model.
    model = Model("model")
    # Initialize the decision variable.
    if type(x_dim) == int and x_dim == 1:
        x = model.addVar(lb=None, ub=None, vtype="C", name="x")
    elif type(x_dim) == int and x_dim > 1:
        x = {}
        for i in range(x_dim):
            x[i] = model.addVar(lb=0, ub=None, vtype="C", name="x(%s)" % (i))
    elif type(x_dim) == tuple or type(x_dim) == list:
        x = {}
        for i in range(x_dim[0]):
            for j in range(x_dim[1]):
                x[i, j] = model.addVar(lb=None, ub=None, vtype="C", name="x(%s, %s)" % (i, j))
    else:
        raise Exception("The dimension of the decision variable is not supported.")
    # Set the time limit.
    model.setRealParam("limits/time", config.time_limit)
    # Add inequality constraints.
    for h in hs:
        model.addCons(h(x) <= 0)
    # Add equality constraints.
    for g in gs:
        model.addCons(g(x) == 0)
    # Encode chance constraint.
    model = ChanceConstraintEncoder(model, x, f, training_Ys, delta, method, omega = omega).encode()
    # Add cost function.
    objective = model.addVar(lb=None, ub=None, vtype="C", name="obj")
    model.addCons(J(x) <= objective)
    model.setObjective(objective, "minimize")
    # Solve the model.
    model.hideOutput()
    time_start = time.time()
    model.optimize()
    time_end = time.time()
    if model.getStatus() == "optimal":
        sol = model.getBestSol()
        if type(x_dim) == int and x_dim == 1:
            return sol[x], time_end - time_start
        elif type(x_dim) == int and x_dim > 1:
            return [sol[x[i]] for i in range(x_dim)], time_end - time_start
        else:
            opt_sol = []
            for i in range(x_dim[0]):
                row = []
                for j in range(x_dim[1]):
                    row.append(sol[x[i, j]])
                opt_sol.append(row)
            return opt_sol, time_end - time_start
    else:
        return model.getStatus(), time_end - time_start  # Denotes error (or infeasibility)