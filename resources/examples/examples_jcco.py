# Import necessary modules.
import sys
import numpy as np
from examples_configuration import system_path
import scipy.stats as stats
from pyscipopt import exp
sys.path.append('resources/')
sys.path.append(system_path)
from resources.solver import solve


def main():
    # Problem setting.
    A = np.array([[3, -12, 2], [-10, 3, 5], [5, 3, -15]])
    N_cons = 3
    def generate_random_vector():
        return stats.t.rvs(10, 0, 1, size=3).tolist()
    def f0(x, Y):
        return A[0][0]*x[0] + A[0][1]*x[1] + A[0][2]*x[2] - Y[0]

    def f1(x, Y):
        return A[1][0]*x[0] + A[1][1]*x[1] + A[1][2]*x[2] - Y[1]

    def f2(x, Y):
        return A[2][0]*x[0] + A[2][1]*x[1] + A[2][2]*x[2] - Y[2]
    def J(x):
        c = np.array([1, 1, 1])
        J = sum(c[i]*x[i] for i in range(N_cons))
        return J
    delta = 0.2
    K = 20
    training_Ys = [generate_random_vector() for i in range(K)]
    f = [f0, f1, f2]
    hs = [lambda x: 0 - x[0], lambda x: 0 - x[1], lambda x: 0 - x[2]]
    gs = []

    # Solution with CPP-KKT using Union-Bound:
    x_kkt_union, time_kkt_union = solve(3, delta, training_Ys, hs, gs, f, J, "CPP-KKT", omega = None, robust = None, epsilon = None, joint_method = "Union")
    print(f"The solution by CPP-KKT with Union Bound is {x_kkt_union} with a solver time of {time_kkt_union}.")
    # Solution with CPP-KKT using Max:
    x_kkt_max, time_kkt_max = solve(3, delta, training_Ys, hs, gs, f, J, "CPP-KKT", omega = None, robust = None, epsilon = None, joint_method = "Max")
    print(f"The solution by CPP-KKT with Max is {x_kkt_max} with a solver time of {time_kkt_max}.")
    # Solution with CPP-MIP using Union-Bound:
    x_mip_union, time_mip_union = solve(3, delta, training_Ys, hs, gs, f, J, "CPP-MIP", omega = None, robust = None, epsilon = None, joint_method = "Union")
    print(f"The solution by CPP-MIP with Union Bound is {x_mip_union} with a solver time of {time_mip_union}.")
    # Solution with CPP-MIP using Max:
    x_mip_max, time_mip_max = solve(3, delta, training_Ys, hs, gs, f, J, "CPP-MIP", omega = None, robust = None, epsilon = None, joint_method = "Max")
    print(f"The solution by CPP-MIP with Max is {x_mip_max} with a solver time of {time_mip_max}.")

if __name__ == "__main__":
    main()