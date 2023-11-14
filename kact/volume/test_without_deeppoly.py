import time
from typing import List, Tuple

import numpy as np

from kact.volume.constants import KRELU_METHODS, KSIGMOID_METHODS, KTANH_METHODS, ACT_FUNCTIONS
from kact.volume.polytope_volume import generate_random_points, is_in_polyhedron
from kact.utils.constraints import generate_one_random_octahedron_projection, constraints_to_vertices_cdd
from kact.utils.lp import get_bounds_of_variables
from kact.utils.functions import sigmoid, tanh

use_fconv = False


def reorder_methods(methods: List[str]) -> List[str]:
    for method in methods:
        if method not in KRELU_METHODS.keys() \
                and method not in KSIGMOID_METHODS.keys() \
                and method not in KTANH_METHODS.keys():
            assert f"The {method} is not supported."

    if "triangle" in methods:
        methods.remove("triangle")
        methods.insert(0, "triangle")

    if "pycdd" in methods:
        methods.remove("pycdd")
        methods.insert(0, "pycdd")

    if "cdd" in methods:
        methods.remove("cdd")
        methods.insert(0, "cdd")
    return methods


def _generate_input_constraints(dim: int) -> np.ndarray:
    print("Project...", end="")
    start = time.time()
    constraints = generate_one_random_octahedron_projection(dim)
    # constraints = CONSTRAINTS1
    # constraints = CONSTRAINTS2
    # constraints = CONSTRAINTS3
    print(f"{time.time() - start:.4f}s", end=" ")
    return constraints


def _get_bounds(constraints: np.ndarray) -> Tuple[List[float], List[float]]:
    print("Get bounds...", end="")
    start = time.time()
    lower_bounds, upper_bounds = get_bounds_of_variables(constraints)
    print(f"{time.time() - start:.4f}s", end=" ")
    return lower_bounds, upper_bounds


def generate_sample_points(points_num: int, lower_bounds: List[float], upper_bounds: List[float],
                           samples_area_shape: str, type_: str) -> np.ndarray:
    func = ACT_FUNCTIONS[type_]
    if type_ == "relu":
        lb, ub = lower_bounds, upper_bounds
    else:
        lower_bounds2 = [func(x) for x in lower_bounds]
        upper_bounds2 = [func(x) for x in upper_bounds]
        lb, ub = lower_bounds + lower_bounds2, upper_bounds + upper_bounds2
    start = time.time()
    print("Generate sample points...", end="")
    sample_points = generate_random_points(samples_area_shape, points_num, lb, ub)
    print(f"{time.time() - start:.4f}s", end=" ")
    return sample_points


def _check_in_convex_hull(type_: str, method: str, cx: np.ndarray, cxy: np.ndarray, exact_vertices: List,
                          lower_bounds: List[float], upper_bounds: [float]):
    # if len(exact_vertices) == 0:
    #     if type_ == "relu" and method in {"pycdd", "cdd"}:
    #         with contextlib.suppress(Exception):
    #             exact_vertices = constraints_to_vertices_cdd(cxy)
    #     else:
    #         exact_vertices = []
    #         f = sigmoid if type_ == "sigmoid" else tanh
    #         vars_num = int((cxy.shape[1] - 1) / 2)
    #         l,u = np.array(lower_bounds), np.array(upper_bounds)
    #         rlu = u - l
    #         points_num = 0
    #         while points_num < 50000:
    #             while True:
    #                 random_point = l + np.random.random((vars_num,)) * rlu
    #
    #                 # random_point = np.hstack((np.ones((1,)), random_point))
    #                 if is_in_polyhedron(np.array([random_point]), cx) > 0:
    #                     random_point = np.hstack((random_point, f(random_point)))
    #                     break
    #             exact_vertices.append(random_point)
    #             points_num+=1
    #         exact_vertices = np.array(exact_vertices)
    #
    # if type_ == "relu" and method not in ("pycdd", "cdd") and len(exact_vertices) != 0:
    #     vertices_in = is_in_polyhedron(exact_vertices[:, 1:], cxy)
    #     print(f"[CONTAIN CONVEXHULL VERTICES]: {vertices_in}/{exact_vertices.shape[0]} ", end="")
    # elif type_ != "relu" and len(exact_vertices) != 0:
    #     vertices_in = is_in_polyhedron(exact_vertices, cxy)
    #     print(f"[CONTAIN CONVEXHULL VERTICES]: {vertices_in}/{exact_vertices.shape[0]} ", end="")

    if len(exact_vertices) == 0:
        exact_vertices = []
        f = sigmoid if type_ == "sigmoid" else tanh
        vars_num = int((cxy.shape[1] - 1) / 2)
        l, u = np.array(lower_bounds), np.array(upper_bounds)
        rlu = u - l
        points_num = 0
        while points_num < 50000:
            while True:
                random_point = l + np.random.random((vars_num,)) * rlu

                # random_point = np.hstack((np.ones((1,)), random_point))
                if is_in_polyhedron(np.array([random_point]), cx) > 0:

                    if type_ == "relu":
                        random_point2 = random_point.copy()
                        random_point2[random_point2 < 0] = 0
                        random_point = np.hstack((random_point, random_point2))
                    else:
                        # p = random_point
                        # ax1 = np.sum(cx[:, 1:] * p, axis=1) + cx[:, 0]
                        # if len(ax1[ax1 < 0]) > 0:
                        #     print("!")
                        #     print(ax1[ax1 < 0])
                        random_point = np.hstack((random_point, f(random_point)))
                        p = random_point
                        ax2 = np.sum(cxy[:, 1:] * p, axis=1) + cxy[:, 0]
                        if len(ax2[ax2 < 0]) > 0:
                            print()
                            print(cx.tolist())
                            raise "The method is not sound."
                    break
            exact_vertices.append(random_point)
            points_num += 1
        exact_vertices = np.array(exact_vertices)

    if len(exact_vertices) != 0:
        vertices_in = is_in_polyhedron(exact_vertices, cxy)
        print(f"[CONTAIN CONVEXHULL VERTICES]: {vertices_in}/{exact_vertices.shape[0]} ", end="")

    return exact_vertices


def _check_in_triangle_relaxation(method: str, constraints: np.ndarray, triangle_relaxations: Optional[np.ndarray]):
    if method == "triangle":
        return constraints
    if method not in {"triangle", "pycdd", "cdd"}:
        try:
            vertices = constraints_to_vertices_cdd(constraints)
            vertices_in = is_in_polyhedron(vertices[:, 1:], triangle_relaxations)
            print(f"[POINTS IN TRIANGLE]: {vertices_in}/{vertices.shape[0]}", end="")
        except Exception:
            print("[POINTS IN TRIANGLE]: Precision error", end="")
    return triangle_relaxations


def example_without_deeppoly(methods: List[str], dim: int, test_times: int, points_num: int = 0,
                             check_vertices: bool = False,
                             samples_area_shape: str = "triangle", type_: str = "relu"):
    assert dim >= 0, "Dimension(dim) should not be negative."
    assert test_times >= 0, "Test time(test_times) should not be negative."
    assert points_num >= 0, "Sameple points number(points_num) should not be negative."
    assert samples_area_shape in {"triangle", "box"}, "Only support triangle or box area shape."
    assert type_ in {"relu", "sigmoid", "tanh"}, "Only support relu, sigmoid and tanh."

    if type_ in {"sigmoid", "tanh"}:
        assert samples_area_shape == "box", "Sigmoid or Tanh only support box area shape."
        # assert check_vertices, "Sigmoid or Tanh does not support check vertices."
    elif check_vertices:
        assert "pycdd" in methods or "cdd" in methods, "pycdd or cdd should in the methods if check vertices."

    methods = reorder_methods(methods)
    methods_dic = METHODS_DIC[type_]

    volumes = []
    constrs_nums = []
    for i in range(test_times):
        print(f"{type_} {dim}D TEST {i + 1}".center(100, "-"))

        constraints = _generate_input_constraints(dim)
        # constraints = np.asarray(
        #     [[5.238269547725489, 1.0, 1.0], [3.50281087305486, 1.0, 0.0], [8.993122459612653, 1.0, -1.0],
        #      [4.270416014985171, 0.0, 1.0], [9.527700377075494, 0.0, -1.0], [5.500761653158344, -1.0, 1.0],
        #      [0.5641739062022022, -1.0, 0.0], [9.643291810617882, -1.0, -1.0]])
        lower_bounds, upper_bounds = _get_bounds(constraints)
        print(lower_bounds)
        print(upper_bounds)
        sample_points = None
        if points_num > 0:
            sample_points = generate_sample_points(points_num, lower_bounds, upper_bounds, samples_area_shape, type_)
        print()

        exact_vertices = []
        triangle_relaxations = None
        for method in methods:
            print(f"{str(method).ljust(10)}", end="")
            start = time.time()
            constrs = methods_dic[method](constraints, lower_bounds, upper_bounds)
            print(f"[TIME]: {time.time() - start:.4f}s ", end="")

            volume = 0
            if sample_points is not None:
                num_in = is_in_polyhedron(sample_points, constrs)
                print(f"[VOLUME]: {num_in}/{points_num} ", end="")
                volume = num_in  # / points_num
            volumes.append(volume)

            print(f"[CONSTRS]: {constrs.shape[0]} ", end="")
            constrs_nums.append(constrs.shape[0])

            if check_vertices:
                exact_vertices = _check_in_convex_hull(type_, method, constraints, constrs,
                                                       exact_vertices, lower_bounds, upper_bounds)

            if "triangle" in methods:
                triangle_relaxations = _check_in_triangle_relaxation(method, constrs, triangle_relaxations)

            print()
    volumes = np.asarray(volumes)
    constrs_nums = np.asarray(constrs_nums)

    print(f"Volumes: {volumes}, Constraints nums: {constrs_nums}")


def test_sigtanh():
    test_times = 100
    points_num = 10000

    dim = 2

    methods = ["sci"]
    if use_fconv:
        methods = ["cdd", "fast", "orthant", "sci", "sciplus"]

    t = time.time()
    example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=True,
                             samples_area_shape="box", type_="sigmoid")
    # example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=True,
    #                          samples_area_shape="box", type_="tanh")
    print("[TOTAL USED TIME]: %0.4f" % (time.time() - t))

    # points_num = 100000
    # dim = 3
    # t = time.time()
    # example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=True,
    #                          samples_area_shape="box", type_="sigmoid")
    # example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=True,
    #                          samples_area_shape="box", type_="tanh")
    # print("[TOTAL USED TIME]: %0.4f" % (time.time() - t))
    #
    # dim = 4
    # methods = ["sci", "sciplus"]
    # if use_fconv:
    #     methods = ["fast", "orthant", "sci", "sciplus"]
    # t = time.time()
    # example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=True,
    #                          samples_area_shape="box", type_="sigmoid")
    # example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=True,
    #                          samples_area_shape="box", type_="tanh")
    # print("[TOTAL USED TIME]: %0.4f" % (time.time() - t))
    #
    # dim = 5
    # # methods = ["mlf", "mlfs"]
    # t = time.time()
    # example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=False,
    #                          samples_area_shape="box", type_="sigmoid")
    # example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=False,
    #                          samples_area_shape="box", type_="tanh")
    # print("[TOTAL USED TIME]: %0.4f" % (time.time() - t))
    #
    # dim = 6
    # # methods = ["mlf", "mlfs"]
    # t = time.time()
    # example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=False,
    #                          samples_area_shape="box", type_="sigmoid")
    # example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=False,
    #                          samples_area_shape="box", type_="tanh")
    # print("[TOTAL USED TIME]: %0.4f" % (time.time() - t))


def test_relu():
    test_times = 1
    points_num = 10000

    dim = 2
    # methods = ["mlf", "mlfsbeta", "mlfs", "mlfss", "pycdd", "triangle"]
    methods = ["pycdd", "mlf", "mlfs", "mlfbig", "mlfbig2"]
    t = time.time()
    example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=True,
                             samples_area_shape="triangle", type_="relu")
    print("[TOTAL USED TIME]: %0.4f " % (time.time() - t))

    dim = 3
    t = time.time()
    example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=True,
                             samples_area_shape="triangle", type_="relu")
    print("[TOTAL USED TIME]: %0.4f" % (time.time() - t))

    dim = 4
    methods = ["mlf", "mlfbig", "mlfbig2"]
    t = time.time()
    example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=False,
                             samples_area_shape="triangle", type_="relu")
    print("[TOTAL USED TIME]: %0.4f" % (time.time() - t))
    #
    # dim = 5
    # methods = ["mlf", "mlfs", "mlfbig", "mlfbig2"]
    # t = time.time()
    # example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=False,
    #                          samples_area_shape="triangle", type_="relu")
    # print("[TOTAL USED TIME]: %0.4f" % (time.time() - t))
    #
    # dim = 6
    # methods = ["mlf", "mlfs", "mlfbig", "mlfbig2"]
    # t = time.time()
    # example_without_deeppoly(methods, dim, test_times, points_num, check_vertices=False,
    #                          samples_area_shape="triangle", type_="relu")
    # print("[TOTAL USED TIME]: %0.4f" % (time.time() - t))


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, threshold=np.inf, linewidth=250)
    test_sigtanh()
    # test_relu()
