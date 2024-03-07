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
    saa_omega = 0.03

    # Solution with CPP-KKT:
    print("Solving the problem with CPP-KKT:")
    solution_kkt, time_kkt = solve(1, delta, training_Ys, hs, gs, f, J, "CPP-KKT", omega = None, robust = False, epsilon = None, joint_method = None)
    print(f"The solution by CPP-KKT is {solution_kkt} with a solver time of {time_kkt}.")
    # Solution with CPP-MIP:
    print("Solving the problem with CPP-MIP:")
    solution_mip, time_mip = solve(1, delta, training_Ys, hs, gs, f, J, "CPP-MIP", omega = None, robust = False, epsilon = None, joint_method = None)
    print(f"The solution by CPP-MIP is {solution_mip} with a solver time of {time_mip}.")
    # Solution with SA:
    print("Solving the problem with SA:")
    solution_sa, time_sa = solve(1, delta, training_Ys, hs, gs, f, J, "SA", omega = None, robust = False, epsilon = None, joint_method = None)
    print(f"The solution by CPP-SA is {solution_sa} with a solver time of {time_sa}.")
    # Solution with SAA:
    print("Solving the problem with SAA:")
    solution_saa, time_saa = solve(1, delta, training_Ys, hs, gs, f, J, "SAA", omega = saa_omega, robust = False, epsilon = None, joint_method = None)
    print(f"The solution by CPP-SAA is {solution_saa} with a solver time of {time_saa}.")


if __name__ == "__main__":
    main()