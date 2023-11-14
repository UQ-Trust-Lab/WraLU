"""
A function for calculating the exact convex hull of a k-ReLU group by pycddlib.
"""
import itertools
from typing import List

import cdd
import numpy as np


def krelu_with_pycdd(constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float]) -> np.ndarray:
    """
    This function calculates the exact convex hull of a k-ReLU group by pycddlib.

    :param constraints: The constraints of the K-ReLU group. The shape of the constraints should be (m, n+1), where m is
    the number of constraints, and n is the number of variables. The last column of the constraints is the constant
    term.
    :param lower_bounds: The lower bounds of the variables.
    :param upper_bounds: The upper bounds of the variables.
    :return: The exact convex hull of the K-ReLU group.
    """
    assert constraints.shape[1] - 1 == len(lower_bounds) == len(upper_bounds), \
        "constraints, lower_bounds and the upper_bounds should have the same number of variables."
    assert all(lb < 0 for lb in lower_bounds), "All lower bound should be negative."
    assert all(ub > 0 for ub in upper_bounds), "All upper bound should be positive."

    try:
        h_repr = _cal_convex_hull(constraints, "float")
    except Exception:
        h_repr = _cal_convex_hull(constraints, "fraction")
    return np.asarray(h_repr)


def _cal_convex_hull(constraints: np.ndarray, number_type: str):
    """
    Calculate the convex hull of the k-ReLU group.
    """
    # Process constraints.
    h_repr = cdd.Matrix(constraints, number_type=number_type)
    h_repr.rep_type = cdd.RepType.INEQUALITY

    # Calculate the vertices of the convex hull.
    vertices = get_vertices_of_each_orthant(h_repr)
    v_repr = [[1] + r + [max(x, 0) for x in r] for r in vertices]
    v_repr = cdd.Matrix(v_repr, number_type="fraction")
    v_repr.rep_type = cdd.RepType.GENERATOR

    convex_hull = cdd.Polyhedron(v_repr)
    h_repr = [[float(v) for v in c] for c in convex_hull.get_inequalities()]
    return h_repr


def get_vertices_of_each_orthant(h_repr) -> List[List]:
    """
    Get vertices in each orthant, which is all the vertices including the vertices of the polyhedron and the
    intersection points with each axis.

    Reference: krelu.py of ERAN

    :param h_repr: The H-representation of the polyhedron.
    """
    var_num = len(h_repr[0]) - 1
    vertices = []
    # Get the vertices of the polyhedron.
    for polarity in itertools.product([-1, 1], repeat=var_num):
        new_mat = h_repr.copy()
        for i in range(var_num):
            constraint = [polarity[i] if j == i + 1 else 0 for j in range(var_num + 1)]
            new_mat.extend([constraint])
        new_mat.canonicalize()  # Remove redundant h constraints.
        v_repr = cdd.Polyhedron(new_mat).get_generators()
        vertices += [list(v[1:]) for v in v_repr]

    return vertices
