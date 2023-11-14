import contextlib
import itertools
import math
import random
import warnings
from typing import List

import cdd
import numpy as np

from .functions import sigmoid, tanh

with contextlib.suppress(Exception):
    import matplotlib.pyplot as plt


def reduce_octahedron_constraints(constraints: np.ndarray, lower_bounds: List[float],
                                  upper_bounds: List[float]) -> np.ndarray:
    warnings.warn("This method is not ready.")
    vars_num = len(lower_bounds)
    remained_rows = []
    distances = []
    for i, constr in enumerate(constraints):
        # point = np.asarray([1.] + [upper_bounds[j] if constr[j + 1] < 0 else lower_bounds[j] for j in range(vars_num)])
        point = np.asarray([1.] + [lower_bounds[j] if constr[j + 1] < 0 else upper_bounds[j] for j in range(vars_num)])
        distance = abs(np.dot(constr, point)) / math.sqrt(sum(abs(constr[1:])))
        distances.append(distance)

    mean = np.mean(distances)
    std = np.std(distances)
    if vars_num <= 4:
        distance_threshold = mean + 0.3 * std
    elif vars_num == 5:
        distance_threshold = mean + 0.2 * std
        # distance_threshold = mean + 0.1 * std
    elif vars_num == 6:
        distance_threshold = mean - 0.2 * std
        # distance_threshold = mean + 0.1 * std
    elif vars_num == 7:
        distance_threshold = mean - 1.0 * std
    elif vars_num == 8:
        distance_threshold = mean - 1.5 * std
    elif vars_num == 9:
        distance_threshold = mean - 2.1 * std
    else:
        distance_threshold = mean - 2.55 * std

    max_remained_num = 100
    for i, distance in enumerate(distances):
        if distance < distance_threshold:
            remained_rows.append(i)
        if len(remained_rows) > max_remained_num:
            break

    constraints = constraints[remained_rows, :]

    return constraints


def constraints_to_vertices_cdd(constraints: np.ndarray) -> np.ndarray:
    constraints = constraints.tolist()
    # try:
    h_repr = cdd.Matrix(constraints, number_type="float")
    h_repr.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(h_repr)
    v_repr = poly.get_generators()
    # except:
    #     h_repr = cdd.Matrix(constraints, number_type="fraction")
    #     h_repr.rep_type = cdd.RepType.INEQUALITY
    #     poly = cdd.Polyhedron(h_repr)
    #     v_repr = poly.get_generators()
    return np.asarray(v_repr)


def normalize_constraints(constraints: np.ndarray):
    return np.dot(np.diag(1 / np.amax(abs(constraints), axis=1)), constraints)


def remove_repeated_constraints(constraints: np.ndarray, tol: float = 1e-8):
    warnings.warn("This method is not ready.")
    constraints = normalize_constraints(constraints)
    deleted_rows = []
    for r1 in range(constraints.shape[0]):
        deleted_rows.extend(r1 for r2 in range(r1 + 1, constraints.shape[0]) \
                            if sum(abs(constraints[r1, :] - constraints[r2, :])) < tol)

    constraints = np.delete(constraints, deleted_rows, axis=0)

    return constraints


def print_ndarray(constraints: np.ndarray):
    output = "-" * 50
    output += "\nshape=" + str(constraints.shape)
    for constraint in constraints:
        ss = ' '.join(str(constraint).split())
        ss = ss[1:-1].strip()
        ss = ss.split(" ")
        output += "\n ["
        for s in ss[:-1]:
            output += format(float(s), ".4f").rjust(7) + "  "
        output += format(float(ss[-1]), ".4f").rjust(7) + "]"
    output += "\n" + "-" * 50
    print(output)


def plot_constraint(constraints: np.ndarray, func: str):
    # plt.style.use('_mpl-gallery')

    # make data
    x = np.linspace(-10, 10, 100)

    fig, ax = plt.subplots()
    for i in range(constraints.shape[0]):
        c = constraints[i]
        if c[2] != 0:
            y = (c[0] + c[1] * x) / -c[2]
            ax.plot(x, y, color="green", linewidth=2.0)
    if func == "sigmoid":
        y = sigmoid(x)
        ax.plot(x, y, color="red")
    elif func == "tanh":
        y = tanh(x)
        ax.plot(x, y, color="red")
    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8), ylim=(0, 8), yticks=np.arange(1, 8))

    plt.show()
