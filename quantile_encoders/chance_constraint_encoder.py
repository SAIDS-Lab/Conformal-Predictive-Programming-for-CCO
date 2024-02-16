"""
In this file, we write different quantile encoders for the chance constraint.
"""

# Import necessary modules.
import numpy as np
from pyscipopt import quicksum
import config

# Hyperparameter setting:
np.random.seed(config.config_seed)

class ChanceConstraintEncoder:
    """
    Encoding the chance constraint.
    """
    def __init__(self, model, x, f, training_ys, delta, method, omega = None):
        """
        Initialize the encoder and perform encoding automatically.

        :param model: the model in which the encoded chance constraint to be added.
        :param x: the decision variable.
        :param f: the function corresponding to the chance constraint, should be upper bounded by 0.
        :param training_ys: the training data Y^{(1)}, ..., Y^{(K)}.
        :param delta: the expected miscoverage rate.
        :param method: the encoding method. Choices include "SA", "SAA", "CPP-KKT", and "CPP-MIP".
        :param omega: the omega parameter for SAA.
        """
        # Initialize fields.
        self.model, self.x, self.f, self.training_ys, self.delta, self.method, self.omega = model, x, f, training_ys, delta, method, omega
        self.K = len(self.training_ys)
        # Check if the method selected is correct.
        if method not in ["SA", "SAA", "CPP-KKT", "CPP-MIP"]:
            raise Exception("The given encoding method is not supported.")
        # Check for omega.
        if self.method == "SAA":
            if self.omega is None:
                raise Exception("The omega parameter is not set for SAA.")
            if self.omega <= 0 or self.omega >= 1:
                raise Exception("The omega parameter should be in the range (0, 1).")
        # Add the encoded constraint.
        if self.method == "SA":
            self.__encode_with_sa()
        elif self.method == "SAA":
            self.__encode_with_saa()
        elif self.method == "CPP-KKT":
            self.__encode_with_cpp_kkt()
        else:
            self.__encode_with_cpp_mip()

    def __encode_with_sa(self):
        """
        Encode the chance constraint via SA.
        """
        for i in range(self.K):
            self.model.addCons(self.f(self.x, self.training_ys[i]) <= 0)

    def __encode_with_saa(self):
        """
        Encode the chance constraint via SAA.
        """
        # Initialize binary variables.
        zs = {}
        for i in range(self.K):
            zs[i] = self.model.addVar(vtype="B", name="zs(%s)" % (i))
        # Add constraints.
        for i in range(self.K):
            self.model.addCons(self.f(self.x, self.training_ys[i]) <= config.M * (1 - zs[i]))
            self.model.addCons(self.f(self.x, self.training_ys[i]) >= config.zeta + (config.m - config.zeta) * zs[i])
        self.model.addCons(quicksum(zs[i] for i in range(self.K)) >= self.K * (1 - self.omega))

    def __encode_with_cpp_kkt(self):
        """
        Encode the chance constraint via CPP with KKT reformulation.
        """
        alpha = (1 + 1 / self.K) * (1 - self.delta)
        # Check for the correct training data size.
        if (self.K < np.ceil((self.K + 1) * (1 - self.delta))):
            raise Exception("Training dataset not large enough.")
        # Initialize variables.
        q = self.model.addVar(lb = None, ub = None, vtype = "C", name = "q")
        gammas, lambdas, betas, e_minuses, e_pluses = {}, {}, {}, {}, {}
        for i in range(self.K):
            gammas[i], lambdas[i], betas[i], e_minuses[i], e_pluses[i] = [self.model.addVar(lb=None, ub=None, vtype="C") for j in range(5)]
        # Define the constraints.
        self.model.addCons(q <= 0)
        for i in range(self.K):
            self.model.addCons(alpha + gammas[i] - lambdas[i] == 0)
        for i in range(self.K):
            self.model.addCons(1 - alpha - gammas[i] - betas[i] == 0)
        summation = 0
        for i in range(self.K):
            summation += gammas[i]
        self.model.addCons(summation == 0)
        for i in range(self.K):
            self.model.addCons(e_pluses[i] - e_minuses[i] - self.f(self.x, self.training_ys[i]) + q == 0)
            self.model.addCons(e_minuses[i] >= 0)
            self.model.addCons(e_pluses[i] >= 0)
            self.model.addCons(lambdas[i] >= 0)
            self.model.addCons(betas[i] >= 0)
            self.model.addCons(lambdas[i] * e_pluses[i] == 0)
            self.model.addCons(betas[i] * e_minuses[i] == 0)

    def __encode_with_cpp_mip(self):
        """
        Encode the chance constraint via CPP with MIP reformulation.
        """
        num_largerthan0_ceil = int(np.ceil((self.K + 1) * (1 - self.delta)))
        # Check for the correct training data size.
        if (self.K < np.ceil((self.K + 1) * (1 - self.delta))):
            raise Exception("Training dataset not large enough.")
        # Initialize variables.
        zs = {}
        for i in range(self.K):
            zs[i] = self.model.addVar(vtype="B", name="zs(%s)" % (i))
        # Add constraints.
        for i in range(self.K):
            self.model.addCons(self.f(self.x, self.training_ys[i]) <= config.M * (1 - zs[i]))
            self.model.addCons(self.f(self.x, self.training_ys[i]) >= config.zeta + (config.m - config.zeta) * zs[i])
        self.model.addCons(quicksum(zs[i] for i in range(self.K)) >= num_largerthan0_ceil)