# Import necessary modules.
import sys
import numpy as np
from examples_configuration import system_path
from pyscipopt import exp
sys.path.append('resources/')
sys.path.append(system_path)
from resources.solver import solve


def main():
    # Experimental setting:
    hs = [lambda x: x ** 3 + 20]
    gs = []
    f = lambda x, Y: exp(x) * (50 * Y) - 5
    J = lambda x: (x ** 3) * exp(x)
    K = 100
    training_Ys = [np.random.exponential(3) for i in range(K)]
    delta = 0.05
    set_epsilon = 0.01

    # Solution with CPP-KKT:
    print("Solving the problem with CPP-KKT:")
    solution_kkt, time_kkt = solve(1, delta, training_Ys, hs, gs, f, J, "CPP-KKT", omega = None, robust = True, epsilon = set_epsilon, joint_method = None)
    print(f"The solution by CPP-KKT is {solution_kkt} with a solver time of {time_kkt}.")
    # Solution with CPP-MIP:
    print("Solving the problem with CPP-MIP:")
    solution_mip, time_mip = solve(1, delta, training_Ys, hs, gs, f, J, "CPP-MIP", omega = None, robust = True, epsilon = set_epsilon, joint_method = None)
    print(f"The solution by CPP-MIP is {solution_mip} with a solver time of {time_mip}.")


if __name__ == "__main__":
    main()