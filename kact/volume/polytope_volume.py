from typing import Optional, List, Tuple

import numpy as np


class VolumeEstimator:
    def __init__(self, constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float]):
        self.constraints = constraints
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.random_points = None

    def estimate(self, estimate_method, random_points: Optional[np.ndarray] = None, step_length=0.1):
        if estimate_method == "random":
            points_in = self._estimate_by_random_points(random_points)
            points_num = random_points.shape[0]
        elif estimate_method == "grid":
            points_in, points_num = self._estimate_by_grid(step_length)
        else:
            raise ValueError("Unknown estimation method")
        return points_in, points_num

    def _estimate_by_random_points(self, points: np.ndarray, tol=1e-8) -> int:
        constraints = self.constraints
        max_points_num = 100000000
        points_num = points.shape[0]
        points_in = 0
        if points_num < max_points_num:
            step_length = 1000

            for start in range(0, points_num, step_length):
                end = min(points_num, start + step_length)
                p = points[start:end]
                ax = constraints[:, 1:] @ p.T + constraints[:, :1]
                points_in += np.count_nonzero(np.all(ax > -tol, axis=0))
        else:
            for p in points:
                if np.all(constraints[:, 1:] @ p + constraints[:, :1] > 0):
                    points_in += 1
        return points_in

    def _estimate_by_grid(self, step_length=0.1, tol=1e-8) -> Tuple[int, int]:
        constraints = self.constraints
        lower_bounds = self.lower_bounds
        upper_bounds = self.upper_bounds

        grid_points = np.mgrid[tuple(slice(lower, upper + step_length, step_length) for lower, upper in
                                     zip(lower_bounds, upper_bounds))].reshape(len(lower_bounds), -1).T

        ax = constraints[:, 1:] @ grid_points.T + constraints[:, :1]

        points_in = np.count_nonzero(np.all(ax > -tol, axis=0))
        points_num = grid_points.shape[0]

        return points_in, points_num
