"""
In this file, we configure the project via setting the hyperparameters.
"""

# Seed
config_seed = 1234

# Hyperparameters for the MIP encoding for Chance Constraint Reformulation.
M = 10000
m = -M
zeta = 0.00001

# Hyperparameters for the solvers.
time_limit = 100