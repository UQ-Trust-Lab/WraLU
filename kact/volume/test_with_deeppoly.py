import os
import sys

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, '..')
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../ELINA/python_interface/tests')
from konvexappro.deeppoly_layer import generate_one_deeppoly_layer
from konvexappro.klayer import KLayer
from konvexappro.test_without_deeppoly import *


def _build_model(type_: str, dim: int) -> KLayer:
    specLB, specUB = -np.ones(dim), np.ones(dim)
    # weights, bias = np.random.random((dim, dim)) * 2 - 1, np.random.random((dim, 1)) * 2 - 1
    weights, bias = np.random.random((dim, dim)) * 2 - 1, np.random.random((dim, 1)) * 2 - 1
    man, element, lower_bounds, upper_bounds = generate_one_deeppoly_layer(type_, dim, specLB, specUB, weights, bias)
    # limit = 4 if type_ == "Sigmoid" else 3
    # while True:
    #     weights, bias = np.random.random((dim, dim)) * 2 - 1, np.random.random((dim, 1)) * 2 - 1
    #     man, element, lower_bounds, upper_bounds = generate_one_deeppoly_layer(type_, dim, specLB, specUB, weights,
    #                                                                            bias)
    #     if all(ub - lb >= 0.1 and lb <= limit and ub >= -limit and lb < 0 < ub for lb, ub in
    #        zip(lower_bounds, upper_bounds)):
    #         break
    #     elina_abstract0_free(man, element)

    return KLayer(man, element, type_, lower_bounds, upper_bounds)


def example_one_layer_with_deeppoly(type_: str, methods: List, dim: int, points_num: int, group_size: int,
                                    overlap_num: int, test_times: int = 1,
                                    check_vertices: bool = False, samples_area_shape: str = "triangle"):
    assert dim >= 0, "Dimension(dim) should not be negative."
    assert test_times >= 0, "Test time(test_times) should not be negative."
    assert points_num >= 0, "Sameple points number(points_num) should not be negative."
    assert samples_area_shape in {"triangle", "box"}, "Only support triangle or box area shape."
    assert type_ in {"Relu", "Sigmoid", "Tanh"}, "Only support relu, sigmoid and tanh."

    if type_ in {"Sigmoid", "Tanh"}:
        assert samples_area_shape == "box", "Sigmoid or Tanh only support box area shape."
        assert not check_vertices, "Sigmoid or Tanh does not support check vertices."
    elif check_vertices:
        assert "pycdd" in methods or "cdd" in methods, "pycdd or cdd should in the methods if check vertices."

    methods = reorder_methods(methods)

    volumes_list = []
    constrs_nums_list = []
    ts_list = []
    for i in range(test_times):
        print(f"----------TEST EXAMPLE {i + 1}----------")

        print("Build...", end="", flush=True)
        start = time.time()
        klayer = _build_model(type_, dim)
        print(f"{time.time() - start:.4f}s", end=" ")
        lower_bounds = klayer.lb
        upper_bounds = klayer.ub

        print("Group...", end="", flush=True)
        start = time.time()
        klayer.group(1, group_size=group_size, overlap_num=overlap_num)
        print(f"\n{klayer.groups}")
        print(f"{time.time() - start:.4f}s", end=" ")

        print("Project...", end="", flush=True)
        start = time.time()
        klayer.project()
        print(f"{time.time() - start:.4f}s", end=" ")

        sample_points = None
        if points_num > 0:
            sample_points = generate_sample_points(points_num, lower_bounds, upper_bounds, samples_area_shape,
                                                   type_.lower())
        print()
        volumes = []
        constrs_nums = []
        ts = []
        for method in methods:
            print(f"{str(method).ljust(10)}", end="", flush=True)
            t = time.time()
            klayer.convex_approx(method)
            t = time.time() - t
            print("[USED TIME]: %f4 " % t, end="")
            ts.append(t)

            constrs = klayer.convex_approximation

            volume = 0
            if sample_points is not None:
                num_in = is_in_polyhedron(sample_points, constrs)
                print(f"[VOLUME]: {num_in}/{points_num} ", end="")
                volume = num_in  # / points_num
            volumes.append(volume)

            print(f"[CONSTRS]: {constrs.shape[0]} ", end="")
            constrs_nums.append(constrs.shape[0])
            print()
        volumes_list.append(volumes)
        constrs_nums_list.append(constrs_nums)
        ts_list.append(ts)
        print()

    return volumes_list, constrs_nums_list, ts_list


def example(type_: str, methods: List[str], dim: int, group_size: int, overlap_num: int, points_num: int,
            test_times: int, check_vertices: bool = False, sample_points_area_shape: str = "box"):
    for method in methods:
        file_path1 = f"./results_st/volume_{method}_d{dim}g{group_size}o{overlap_num}p{points_num}{sample_points_area_shape}.txt"
        file_path2 = f"./results_st/constrs_num_{method}_d{dim}g{group_size}o{overlap_num}p{points_num}{sample_points_area_shape}.txt"
        file_path3 = f"./results_st/used_time_{method}_d{dim}g{group_size}o{overlap_num}p{points_num}{sample_points_area_shape}.txt"
        with open(file_path1, "a") as file:
            file.write("------------")
        with open(file_path2, "a") as file:
            file.write("------------")
        with open(file_path3, "a") as file:
            file.write("------------")

    v, c, t = example_one_layer_with_deeppoly(type_, methods, dim, points_num,
                                              group_size=group_size, overlap_num=overlap_num,
                                              check_vertices=check_vertices, test_times=test_times,
                                              samples_area_shape=sample_points_area_shape)
    v, c, t = np.asarray(v), np.asarray(c), np.asarray(t)

    vm = np.mean(v, axis=0)
    cm = np.mean(c, axis=0)
    tm = np.mean(t, axis=0)
    v = np.vstack((v, vm))
    c = np.vstack((c, cm))
    t = np.vstack((t, tm))

    for i, method in enumerate(methods):
        file_path1 = f"./results_st/volume_{method}_d{dim}g{group_size}o{overlap_num}p{points_num}{sample_points_area_shape}.txt"
        file_path2 = f"./results_st/constrs_num_{method}_d{dim}g{group_size}o{overlap_num}p{points_num}{sample_points_area_shape}.txt"
        file_path3 = f"./results_st/used_time_{method}_d{dim}g{group_size}o{overlap_num}p{points_num}{sample_points_area_shape}.txt"

        with open(file_path1, "a") as file:
            file.write(str(v[:, i]))
        with open(file_path2, "a") as file:
            file.write(str(c[:, i]))
        with open(file_path3, "a") as file:
            file.write(str(t[:, i]))


def example_one_group1(type_: str, test_times: int, check_vertices: bool = False):
    sample_points_area_shape = "triangle" if type_ == "Relu" else "box"

    methods = ["cdd", "fast", "orthant", "mlf", "mlfs"]
    # methods = ["cdd"]
    example(type_, methods, 2, 2, 0, 10000, test_times,
            check_vertices=check_vertices, sample_points_area_shape=sample_points_area_shape)

    # methods = ["fast", "orthant", "mlf", "mlfs"]
    # example(type_, methods, 3, 3, 0, 100000, test_times,
    #         check_vertices=check_vertices, sample_points_area_shape=sample_points_area_shape)

    # methods = ["fast", "mlf", "mlfs"]
    # example(type_,  methods, 4, 4, 0, 10000, test_times,
    #         check_vertices=check_vertices, sample_points_area_shape=sample_points_area_shape)

    # methods = ["mlf", "mlfs"]
    # example(type_, methods, 5, 5, 0, 10000, test_times,
    #         check_vertices=check_vertices, sample_points_area_shape=sample_points_area_shape)
    # example(type_,  methods, 6, 6, 0, 100000, test_times,
    #         check_vertices=check_vertices, sample_points_area_shape=sample_points_area_shape)


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, threshold=np.inf, linewidth=250)
    test_times = 10
    for _ in range(1):
        # example_one_group1("Relu", test_times)
        example_one_group1("Tanh", test_times)
