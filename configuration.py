"""
In this file, we configure the project via setting the hyperparameters.
"""

# System path.
system_path = '/Users/yiqizhao/Dropbox/Mac/Documents/Conformal-Predictive-Programming-for-CCO/' # Chance this to your own path to run the codes.

# Seed
config_seed = 1234

# Hyperparameters for the MIP encoding for Chance Constraint Reformulation.
M = 10000
m = -M
zeta = 0.00001

# Hyperparameters for the solvers.
time_limit = 100