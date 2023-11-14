from typing import Optional, List, Tuple

import numpy as np

from kact.volume.sample_points_generator import SamplePointsGenerator


class VolumeEstimator:
    def __init__(self, constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float]):
        self.constraints = constraints
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.random_points = None

    def estimate(self, estimate_method,
                 random_points: Optional[np.ndarray] = None,
                 step_length=0.1,
                 points_num: Optional[int] = None,
                 lower_bounds: Optional[List] = None,
                 upper_bounds: Optional[List] = None,
                 max_points_in: int = None,
                 sample_area_shape: str = None) -> Tuple[int, int]:
        if estimate_method == "random":
            points_in, points_num = self._estimate_by_random_points(random_points, points_num, lower_bounds, upper_bounds,
                                                        sample_area_shape, max_points_in)
        elif estimate_method == "grid":
            points_in, points_num = self._estimate_by_grid(step_length)
        else:
            raise ValueError("Unknown estimation method")
        return points_in, points_num

    def _estimate_by_random_points(self, points: Optional[np.ndarray] = None,
                                   points_num: Optional[int] = None,
                                   lower_bounds: Optional[List] = None,
                                   upper_bounds: Optional[List] = None,
                                   sample_area_shape: str = None,
                                   max_points_in: int = None,
                                   tol=1e-8) -> Tuple[int, int]:
        constraints = self.constraints

        sample_points_generator = None
        if points is not None:
            points_num = points.shape[0]
        elif max_points_in is not None:
            points_num = int(1e10)

        if points is None:
            sample_points_generator = SamplePointsGenerator(lower_bounds, upper_bounds)

        points_in = 0
        points_num2 = 0
        step_length = 10000
        for start in range(0, points_num, step_length):
            end = min(points_num, start + step_length)
            if sample_points_generator is not None:
                p = sample_points_generator.generate(end - start, sample_area_shape)
            else:
                p = points[start:end]
            points_num2 += p.shape[0]
            ax = constraints[:, 1:] @ p.T + constraints[:, :1]
            points_in += np.count_nonzero(np.all(ax > -tol, axis=0))
            if max_points_in is not None and points_in > max_points_in:
                break

        return points_in, points_num2

    def _estimate_by_grid(self, step_length=0.1, tol=1e-8) -> Tuple[int, int]:
        constraints = self.constraints

        lower_bounds = self.lower_bounds * 2
        upper_bounds = self.upper_bounds * 2
        assert len(lower_bounds) == len(upper_bounds), "Lower and upper bounds must have the same length"
        assert len(lower_bounds) == constraints.shape[1] - 1, \
            (f"Lower and upper bounds must have the same length as the number of constraints, "
             f"but got {len(lower_bounds)} and {len(upper_bounds)} for lower and upper bounds, "
             f"and {constraints.shape[1] - 1} for constraints")

        grid_points = np.mgrid[tuple(slice(lower, upper + step_length, step_length) for lower, upper in
                                     zip(lower_bounds, upper_bounds))].reshape(len(lower_bounds), -1).T

        ax = constraints[:, 1:] @ grid_points.T + constraints[:, :1]

        points_in = np.count_nonzero(np.all(ax >= 0, axis=0))
        points_num = grid_points.shape[0]

        return points_in, points_num
