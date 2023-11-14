from typing import List, Optional

import numpy as np


class SamplePointsGenerator:
    def __init__(self, lower_bounds: List[float], upper_bounds: List[float]):
        assert len(lower_bounds) == len(upper_bounds), "The length of lower bounds and upper bounds must be the same."
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def generate(self, points_num: int, samples_area_shape: str) -> Optional[np.ndarray]:
        assert samples_area_shape in {"box", "triangle"}, f"Shape {samples_area_shape} is not supported."
        if points_num == 0:
            return None
        # print("Generate sample points...", end="")

        if samples_area_shape == "box":
            # lb = self.lower_bounds + [f(x) for x in self.lower_bounds]
            # ub = self.upper_bounds + [f(x) for x in self.upper_bounds]
            lb = self.lower_bounds * 2
            ub = self.upper_bounds * 2
            # lb, ub = self.lower_bounds, self.upper_bounds
            sample_points = self._generate_random_points_in_box(points_num, lb, ub)
        else:
            sample_points = self._generate_random_points_in_triangle(points_num, self.lower_bounds, self.upper_bounds)

        # print(f"{time.time() - start:.4f}s")
        return sample_points

    @staticmethod
    def _generate_random_points_in_box(points_num: int, lower_bounds: List[float], upper_bounds: List[float]
                                       ) -> np.ndarray:
        vars_num = len(lower_bounds)
        r = np.random.random((points_num, vars_num), )
        lbs = np.array(lower_bounds)
        ubs = np.array(upper_bounds)
        r = lbs + r * (ubs - lbs)

        return r

    @staticmethod
    def _generate_random_points_in_triangle(points_num: int, lower_bounds: List[float], upper_bounds: List[float]
                                            ) -> np.ndarray:
        vars_num = len(lower_bounds)
        vector_a = np.tile(np.hstack((np.array([lower_bounds]), np.zeros((1, vars_num)))), (points_num, 1))
        vector_b = np.tile(np.array([upper_bounds]), (points_num, 2))
        t1 = np.random.random((points_num, vars_num))
        t2 = np.random.random((points_num, vars_num))
        bad_para_locations = np.where(t1 + t2 > 1)
        t1[bad_para_locations] = 1 - t1[bad_para_locations]
        t2[bad_para_locations] = 1 - t2[bad_para_locations]
        t1 = np.tile(t1, (1, 2))
        t2 = np.tile(t2, (1, 2))
        return vector_a * t1 + vector_b * t2
