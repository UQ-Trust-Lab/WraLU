import itertools
import random
import time

import numpy as np


class InputConstraintsGenerator:

    def __init__(self, dim: int):
        self.dim = dim

    def generate(self, method: str, lower_bound: float = 0, upper_bound: float = 0, constrs_num: int = 0) -> np.ndarray:
        # print("Generate Input Constraints...", end="")

        start = time.time()
        if method == "box+random":
            constriants = np.vstack([
                self.generate_box_constraints(self.dim, lower_bound, upper_bound),
                self.generate_random_constraints(self.dim, lower_bound, upper_bound, constrs_num)
            ])
        elif method == "octahedron":
            constriants = self._generate_octahedron_input_constraints(self.dim)
        else:
            raise ValueError(f"Unknown method: {method}")
        # print(f"{time.time() - start:.4f}s")
        return constriants

    @staticmethod
    def _generate_octahedron_input_constraints(dim: int) -> np.ndarray:
        constraints = []
        for coeffs in itertools.product([-1, 0, 1], repeat=dim):
            if all(c == 0 for c in coeffs):
                continue
            constraint = [random.random() * 10] + [-c for c in coeffs]
            constraints.append(constraint)
        return np.asarray(constraints)

    @staticmethod
    def generate_box_constraints(dim: int, lower_bound: float, upper_bound: float) -> np.ndarray:
        lbs, ubs = [lower_bound] * dim, [upper_bound] * dim
        lb, ub = -np.array(lbs).reshape((-1, 1)), np.array(ubs).reshape((-1, 1))
        v1, v2 = np.identity(dim), -np.identity(dim)
        return np.vstack([np.hstack([lb, v1]), np.hstack([ub, v2])])

    @staticmethod
    def generate_random_constraints(dim: int, lower_bound: float, upper_bound: float, number: int) -> np.ndarray:
        constraints = []
        lower_bound, upper_bound = -1., 1.
        r = upper_bound - lower_bound
        for _ in range(number):
            constraint = [r * random.random() + lower_bound for __ in range(dim + 1)]
            constraint[0] = abs(constraint[0]) # Make sure the bias is positive
            constraints.append(constraint)
        return np.asarray(constraints)
