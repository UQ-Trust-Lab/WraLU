import itertools
import time
from typing import List, Tuple

import numpy as np
from gurobipy import GRB

from kact.utils import get_bounds_of_variables, solve_lp
from kact.constants import METHODS_DIC
from .check_methods import check_methods
from .input_constrs_generator import InputConstraintsGenerator
from .volume_estimator import VolumeEstimator
from .soundness_checker import SoundnessChecker
from .sample_points_generator import SamplePointsGenerator


class VolumeCalculator:
    def __init__(self, func: str, methods: List[str], dim: int = 2, test_times: int = 1,
                 input_constrs_method: str = "octahedron",
                 input_constrs_box_lower_bound: float = 0.0,
                 input_constrs_box_upper_bound: float = 0.0,
                 input_constrs_num: int = 0,
                 volume_method: str = "random",
                 volume_area_shape: str = "box",
                 volume_grid_point_step_length: float = 0.1,
                 volume_random_points_num: int = 0,
                 check_soundness: bool = False,
                 check_soundness_method: str = "random",
                 output_file_path: str = None
                 ):
        assert func in {"relu", "sigmoid", "tanh"}, "Only support relu, sigmoid and tanh."
        assert volume_area_shape in {"triangle", "box"}, "Only support triangle or box area shape."
        assert dim >= 0, "Dimension(dim) should not be negative."
        assert test_times >= 0, "Test time(test_times) should not be negative."
        assert volume_random_points_num >= 0, "Sameple points number(points_num) should not be negative."
        if func in {"sigmoid", "tanh"}:
            assert volume_area_shape == "box", "Sigmoid or Tanh only support box area shape."

        self.func = func
        self.methods = check_methods(methods)
        self.dim = dim
        self.test_times = test_times
        self.input_constrs_method = input_constrs_method
        self.input_constrs_num = input_constrs_num
        self.input_constrs_box_lower_bound = input_constrs_box_lower_bound
        self.input_constrs_box_upper_bound = input_constrs_box_upper_bound
        self.volume_area_shape = volume_area_shape
        self.volume_method = volume_method
        self.volume_grid_point_step_length = volume_grid_point_step_length
        self.volume_random_points_num = volume_random_points_num
        self.check_soundness = check_soundness
        self.check_soundness_method = check_soundness_method
        self.output_file_path = output_file_path
        self._create_output_file()
        self.input_constrts_generator = InputConstraintsGenerator(dim)

    def _create_output_file(self):
        if self.output_file_path is None:
            return

        with open(self.output_file_path, "w") as f:
            f.write(f"func\t{self.func}\n")
            f.write(f"methods\t{self.methods}\n")
            f.write(f"dim\t{self.dim}\n")
            f.write(f"test_times\t{self.test_times}\n")
            f.write(f"input_constrs_method\t{self.input_constrs_method}\n")
            f.write(f"input_constrs_num\t{self.input_constrs_num}\n")
            f.write(f"input_constrs_box_lower_bound\t{self.input_constrs_box_lower_bound}\n")
            f.write(f"input_constrs_box_upper_bound\t{self.input_constrs_box_upper_bound}\n")
            f.write(f"volume_area_shape\t{self.volume_area_shape}\n")
            f.write(f"volume_method\t{self.volume_method}\n")
            f.write(f"volume_grid_point_step_length\t{self.volume_grid_point_step_length}\n")
            f.write(f"volume_random_points_num\t{self.volume_random_points_num}\n")
            f.write(f"check_soundness\t{self.check_soundness}\n")
            f.write(f"check_soundness_method\t{self.check_soundness_method}\n")
            for method in self.methods:
                f.write(f"{method}\t\t\t\t\t")
            f.write('\n')
            for _ in self.methods:
                f.write(f'id\ttime\tpoints_in\tpoints_num\tconstrs_num\t')
            f.write('\n')

    def _record_samples(self, id: int, time: float, volume: float, points_num, constrs_nums: int, is_end: bool = False):
        if self.output_file_path is None:
            return

        with open(self.output_file_path, "a") as f:
            f.write(f"{id}\t{time}\t{volume}\t{points_num}\t{constrs_nums}\t")
            if is_end:
                f.write('\n')

    def calculate(self):
        volumes = []
        constrs_nums = []

        for i in range(self.test_times):
            print(f"{self.func} {self.dim}D TEST {i + 1}".center(100, "-"))

            # Generate input constraints
            input_constrs = self.input_constrts_generator.generate(self.input_constrs_method,
                                                                   self.input_constrs_box_lower_bound,
                                                                   self.input_constrs_box_upper_bound,
                                                                   self.input_constrs_num)
            input_constrs_oct = self._create_octahedron_approximation(input_constrs)

            # Calculate bounds
            lb, ub = self._get_bounds(input_constrs)

            # Generate sample points
            sample_points = self._generate_sample_points(lb, ub)

            real_constrs = None
            # Calculate volumes and constraints nums for each method
            for method in self.methods:
                print(f"{str(method).ljust(10)}", end="")

                # Calculate constraints
                output_constrs, used_time = self._cal_constrs(method, input_constrs, input_constrs_oct, lb, ub)
                constrs_num = output_constrs.shape[0]
                constrs_nums.append(constrs_num)
                if method in {"cdd", "pycdd"}:
                    real_constrs = output_constrs

                # Calculate volume
                volume, points_num = self._cal_volume(self.volume_method, output_constrs, lb, ub, sample_points,
                                                      self.volume_grid_point_step_length)
                volumes.append(volume)

                # Check soundness
                self._check_soundness(method, input_constrs, output_constrs, lb, ub, real_constrs)

                # Record samples
                is_end = method == self.methods[-1]
                self._record_samples(i, used_time, volume, points_num, constrs_num, is_end)

                print()

        volumes = np.asarray(volumes)
        constrs_nums = np.asarray(constrs_nums)

        print(f"Volumes: {volumes}, Constraints nums: {constrs_nums}")

    def _cal_constrs(self, method: str, input_constrs: np.ndarray, input_constrs_oct: np.ndarray, lb: List[float],
                     ub: List[float]) -> Tuple[np.ndarray, float]:
        start = time.time()
        if method in {"cdd", "fast"}:
            output_constrs = METHODS_DIC[self.func][method](input_constrs_oct, lb, ub)
        elif method in {"pycdd"}:
            output_constrs = METHODS_DIC[self.func][method](input_constrs, lb, ub)
        elif method in {"triangle", "quad"}:
            output_constrs = METHODS_DIC[self.func][method](input_constrs, lb, ub)
        else:
            if self.func == "relu":
                output_constrs = METHODS_DIC[self.func][method](input_constrs, lb, ub, add_triangle=True, check=False)
            else:
                output_constrs = METHODS_DIC[self.func][method](input_constrs, lb, ub, add_quadrilateral=True, check=False)
        used_time = time.time() - start
        print(f"[TIME]: {used_time:.4f}s ", end="")
        print(f"[CONSTRS]: {output_constrs.shape[0]} ", end="")
        return output_constrs, used_time

    def _cal_volume(self, method: str, output_constrs: np.ndarray, lb: List[float], ub: List[float],
                    random_points: np.ndarray = None,
                    step_length: float = 0.1) -> Tuple[float, int]:
        volume = 0
        if random_points is not None:
            volume_estimator = VolumeEstimator(output_constrs, lb, ub)
            points_in_num, points_num = volume_estimator.estimate(method,
                                                                  random_points=random_points,
                                                                  step_length=step_length)
            print(f"[VOLUME]: {points_in_num}/{points_num} ", end="")
            volume = points_in_num  # / points_num
            return volume, points_num
        return volume, 0

    def _check_soundness(self, method: str, input_constrs: np.ndarray, output_constrs: np.ndarray,
                         lb: List[float], ub: List[float], real_constrs: np.ndarray):
        if self.check_soundness:
            soundness_checker = SoundnessChecker(input_constrs, output_constrs, lb, ub, self.func)
            soundness_checker.check(self.check_soundness_method, method, real_constraints=real_constrs)

    def _generate_sample_points(self, lb: List[float], ub: List[float]) -> np.ndarray:
        sample_points_generator = SamplePointsGenerator(lb, ub)
        return sample_points_generator.generate(self.volume_random_points_num, self.volume_area_shape, self.func)

    @staticmethod
    def _get_bounds(constraints: np.ndarray) -> Tuple[List[float], List[float]]:
        print("Get bounds...", end="")
        start = time.time()
        lower_bounds, upper_bounds = get_bounds_of_variables(constraints)
        print(f"{time.time() - start:.4f}s")
        return lower_bounds, upper_bounds

    @staticmethod
    def _create_octahedron_approximation(constraints: np.ndarray) -> np.ndarray:
        dim = constraints.shape[1] - 1
        oct_constrs = []
        for coeffs in itertools.product([-1, 0, 1], repeat=dim):
            if all(c == 0 for c in coeffs):
                continue
            obj = np.asarray([0] + list(coeffs))
            _, bias = solve_lp(constraints, obj, GRB.MAXIMIZE)
            constr = [bias] + [-c for c in coeffs]
            oct_constrs.append(constr)

        return np.asarray(oct_constrs)
