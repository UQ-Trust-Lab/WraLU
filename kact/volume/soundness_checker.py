from typing import List, Optional

import numpy as np

from kact import krelu_with_triangle
from kact.utils import constraints_to_vertices_cdd
from kact.constants import ACT_FUNCTIONS


class SoundnessChecker:
    def __init__(self, input_constraints: np.ndarray, output_constraints: np.ndarray,
                 lower_bounds: List[float], upper_bounds: List[float], func: str):
        assert input_constraints.shape[1] == len(lower_bounds) + 1, "Constraints and lower bounds are not match."
        assert input_constraints.shape[1] == len(upper_bounds) + 1, "Constraints and upper bounds are not match."
        assert output_constraints.shape[1] == 2 * input_constraints.shape[1] - 1, \
            "Output constraints and input constraints are not match."
        assert len(lower_bounds) == len(upper_bounds), "Lower bounds and upper bounds are not match."
        assert func in {"relu", "sigmoid", "tanh"}, "Only support relu, sigmoid and tanh."

        self.input_constraints = input_constraints
        self.output_constraints = output_constraints
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.func = func
        self.triangle_relaxations = None
        self.vertices = None
        self.random_points = None

    def check(self, check_method: str, method: str,
              real_constraints: Optional[np.ndarray] = None,
              max_random_points_num: int = 5000):

        if check_method == "exact":
            assert self.func == "relu", "Exact check only support relu."
            self._check_exact_vertices(method, real_constraints)
        elif check_method == "random":
            self._check_random_points(max_random_points_num)

    def _check_exact_vertices(self, method: str, real_constraints: np.ndarray):

        if method == "triangle":
            return None

        if self.triangle_relaxations is None:
            self.triangle_relaxations = krelu_with_triangle(self.input_constraints, self.lower_bounds,
                                                            self.upper_bounds)

        if self.vertices is None:
            try:
                self.vertices = constraints_to_vertices_cdd(real_constraints)
                assert self.vertices.shape[0] > 0, "No vertices."
            except Exception as e:
                print(e)
                return None


        if method not in {"triangle", "pycdd", "cdd"}:
            vertices_in = self.is_in_polyhedron(self.vertices[:, 1:], self.triangle_relaxations)
            print(f"[CONTAIN EXACT VERTICES]: {vertices_in}/{self.vertices.shape[0]}", end="")



    def _check_random_points(self, max_random_points_num: int):
        if self.random_points is None:
            points = []
            vars_num = int((self.output_constraints.shape[1] - 1) / 2)
            l, u = np.array(self.lower_bounds), np.array(self.upper_bounds)
            rlu = u - l
            f = ACT_FUNCTIONS[self.func]
            for _ in range(max_random_points_num):
                while True:
                    point = l + np.random.random((vars_num,)) * rlu
                    if self.is_in_polyhedron(np.array([point]), self.input_constraints) > 0:
                        point = np.hstack((point, f(point)))

                        # Check the soundness of the method.
                        r = np.sum(self.output_constraints[:, 1:] * point, axis=1) + self.output_constraints[:, 0]
                        if np.any(r < 0):
                            print("\n" + str(self.input_constraints.tolist()))
                            raise "The method is not sound."
                        break
                points.append(point)
            points = np.array(points)
            self.random_points = points

        vertices_in = self.is_in_polyhedron(self.random_points, self.output_constraints)
        print(f"[CONTAIN RANDOM POINTS]: {vertices_in}/{max_random_points_num} ", end="")

    @staticmethod
    def is_in_polyhedron(points: np.ndarray, constraints: np.ndarray, step_length=1000, tol=1e-8) -> int:
        points_in = 0
        p = points
        ax = np.dot(constraints[:, 1:], p.T) + np.tile(np.array([constraints[:, 0]]).T, (1, p.shape[0]))
        points_in += np.nonzero(np.all(ax > -tol, 0))[0].shape[0]
        return points_in