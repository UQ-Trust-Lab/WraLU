"""
A function for calculating the convex hull approximation of a k-ReLU group by selective constraints identification (SCI)
method.
"""
import time
from typing import List

import cdd
import numpy as np

from .triangle import krelu_with_triangle


def krelu_with_sci(constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float],
                   add_triangle: bool = False, tol: float = 1e-8, check: bool = False,
                   output_time_cal_vertices=False) -> np.ndarray:
    """
    This function calculates the convex hull approximation of a k-ReLU group by selective constraints identification
    (SCI) method.

    :param constraints: The constraints of the K-ReLU group. The shape of the constraints should be (m, n+1), where m is
    the number of constraints, and n is the number of variables. The last column of the constraints is the constant
    term.
    :param lower_bounds: The lower bounds of the variables.
    :param upper_bounds: The upper bounds of the variables.
    :param add_triangle: Whether to add the triangle relaxation of the k-ReLU group.
    :param tol: The tolerance for using SCI method.
    :return: The convex hull approximation of the K-ReLU group.
    """
    assert constraints.shape[1] - 1 == len(lower_bounds) == len(upper_bounds), \
        f"constraints {constraints.shape[1] - 1}, lower_bounds {len(lower_bounds)} and the " \
        f"upper_bounds {len(upper_bounds)}should have the same number of variables."
    assert all(lb < 0 for lb in lower_bounds), "All lower bound should be negative."
    assert all(ub > 0 for ub in upper_bounds), "All upper bound should be positive."

    constraints = np.array(constraints, dtype=np.float64)

    vars_num = constraints.shape[1] - 1
    # constraints1 = np.hstack((constraints, np.zeros((constraints.shape[0], vars_num))))
    # return constraints1

    # Add the triangle relaxation of the k-ReLU group.
    triangle_constraints = None
    if add_triangle or vars_num == 1:
        triangle_constraints = krelu_with_triangle(constraints, lower_bounds, upper_bounds)
    if vars_num == 1:
        return triangle_constraints

    c = np.ascontiguousarray(constraints, dtype=np.float64)
    time_v = time.time()
    v = cal_vertices(constraints, check=check)
    time_v = time.time() - time_v
    # v[:, 1:] = v[:, 1:]
    x = np.ascontiguousarray(np.transpose(v), dtype=np.float64)

    d = np.matmul(c, x)
    c = cal_betas(c, x, d, tol=tol)
    if output_time_cal_vertices:
        return (np.vstack((c, triangle_constraints)), time_v) if add_triangle else (c, time_v)
    return np.vstack((c, triangle_constraints)) if add_triangle else c


def krelu_with_sciplus(constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float],
                       add_triangle: bool = False, tol: float = 1e-8, check: bool = False,
                       output_time_cal_vertices: bool = False) -> np.ndarray:
    """
    This function calculates the convex hull approximation of a k-ReLU group by selective constraints identification
    (SCI) method in Plus version. Here we use two ordering of the variabls to calculate the SCI constraints. One is the
    original ordering, and the other is the reversed ordering.

    :param constraints: The constraints of the K-ReLU group. The shape of the constraints should be (m, n+1), where m is
    the number of constraints, and n is the number of variables. The last column of the constraints is the constant
    term.
    :param lower_bounds: The lower bounds of the variables.
    :param upper_bounds: The upper bounds of the variables.
    :param add_triangle: Whether to add the triangle relaxation of the k-ReLU group.
    :param tol: The tolerance for using SCI method.
    :return: The convex hull approximation of the K-ReLU group.
    """
    assert constraints.shape[1] - 1 == len(lower_bounds) == len(upper_bounds), \
        f"constraints {constraints.shape[1] - 1}, lower_bounds {len(lower_bounds)} and the " \
        f"upper_bounds {len(upper_bounds)}should have the same number of variables."
    assert all(lb < 0 for lb in lower_bounds), "All lower bound should be negative."
    assert all(ub > 0 for ub in upper_bounds), "All upper bound should be positive."

    constraints = np.array(constraints, dtype=np.float64)
    vars_num = constraints.shape[1] - 1

    triangle_constraints = None
    if add_triangle or vars_num == 1:
        triangle_constraints = krelu_with_triangle(constraints, lower_bounds, upper_bounds)
    if vars_num == 1:
        return triangle_constraints

    constraints = np.ascontiguousarray(constraints, dtype=np.float64)
    time_v = time.time()
    vertices = cal_vertices(constraints, check=check)
    time_v = time.time() - time_v
    vertices = np.ascontiguousarray(np.transpose(vertices), dtype=np.float64)
    m = np.matmul(constraints, vertices)

    # Get two ordering of the variables.
    reversed_order = list(range(vars_num, 0, -1))
    inversed_order = reversed_order + [o + vars_num for o in reversed_order]

    new_constraints = np.empty((0, 2 * vars_num + 1))

    # Calculate the convex approximation by two ordering of the variables.
    for k in range(2):
        c = constraints[:, [0] + reversed_order].copy() if k == 1 else constraints.copy()
        x = vertices[[0] + reversed_order].copy() if k == 1 else vertices.copy()
        d = m.copy()
        c = cal_betas(c, x, d, tol=tol)

        # Recover the ordering of the variables.
        if k == 1:
            c[:, 1:] = c[:, inversed_order]
        new_constraints = np.vstack((new_constraints, c))

    if output_time_cal_vertices:
        return (np.vstack((new_constraints, triangle_constraints)), time_v) if add_triangle else (
        new_constraints, time_v)

    if triangle_constraints is not None:
        return np.vstack((new_constraints, triangle_constraints))

    return new_constraints


def cal_vertices(constraints: np.ndarray, check=False, fraction=False) -> np.ndarray:
    """
    This function calculates the vertices of the polyhedron defined by the constraints. By default, we will use the
    float number type to calculate the vertices. If the number of vertices is too small, we will use the fraction
    type to calculate the vertices.

    :param constraints: The constraints of the polyhedron. The shape of the constraints should be (m, n+1), where m is
    the number of constraints, and n is the number of variables. The last column of the constraints is the constant
    term.
    :param check: Whether to check the number of vertices. If the number of vertices is too small, we will use the
    fraction type to calculate the vertices.
    :param fraction: Whether to directly use the fraction number type to calculate the vertices.
    :return: The vertices of the polyhedron.
    """
    # RISK_THRESHOLD = {3: 28, 4: 250, 5: 3808}
    RISK_THRESHOLD = {3: 46, 4: 382, 5: 3808}

    if fraction:
        mat = cdd.Matrix(constraints, number_type="fraction")
        mat.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(mat)
        vertices = poly.get_generators()
        return np.asarray(vertices, dtype=np.float64)

    vars_num = constraints.shape[1] - 1

    try:
        mat = cdd.Matrix(constraints, number_type="float")
        mat.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(mat)
        vertices = poly.get_generators()

        if check:
            vertices_num = len(vertices)
            if vars_num in RISK_THRESHOLD and vertices_num < RISK_THRESHOLD[vars_num]:
                mat = cdd.Matrix(constraints, number_type="fraction")
                mat.rep_type = cdd.RepType.INEQUALITY
                poly = cdd.Polyhedron(mat)
                vertices = poly.get_generators()
            if vertices_num == len(vertices):
                print(vertices_num, "->", len(vertices), "This is no risk.")

    except Exception:
        mat = cdd.Matrix(constraints, number_type="fraction")
        mat.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(mat)
        vertices = poly.get_generators()
    vertices = np.asarray(vertices, dtype=np.float64)

    assert all(vertices[:, 0] == 1), "The first column of the vertices should be 1."
    return vertices


def cal_betas(c: np.ndarray, x: np.ndarray, d: np.ndarray, tol: float = 1e-8) \
        -> np.ndarray:
    """
    This function calculates the beta1 and beta2 for the convex approximation of the K-ReLU group.

    :param c: The constraints of the polyhedron. The shape of the constraints should be (m, n+1), where m is
    the number of constraints, and n is the number of variables. The last column of the constraints is the constant
    term.
    :param x: The vertices of the polyhedron. The shape of the vertices should be (n+1, k), where n is the number of
    variables, and k is the number of vertices.
    :param d: The matrix of the constraints. The shape of the matrix should be (m, k), where m is the number of
    constraints, and k is the number of vertices.
    :param tol: The tolerance of the calculation.
    :return: The constraints of the convex approximation of the K-ReLU group.
    """
    vars_num = c.shape[1] - 1

    x_greater_zero, x_smaller_zero = (x > 0.0), (x < -0.0)

    y = x.copy()
    y[y < 0.0] = 0.0

    for i in range(1, vars_num + 1):
        vertices_greater_zero, vertices_smaller_zero = x_greater_zero[i], x_smaller_zero[i]
        beta1 = beta2 = np.zeros((c.shape[0], 1), dtype=np.float64)

        if np.any(vertices_greater_zero):
            beta1 = d[:, vertices_greater_zero] / x[i, vertices_greater_zero]
            beta1 = np.max(-beta1, axis=1).reshape((-1, 1))

        if np.any(vertices_smaller_zero):
            beta2 = d[:, vertices_smaller_zero] / x[i, vertices_smaller_zero]
            beta2 = np.max(beta2, axis=1).reshape((-1, 1))

        c = np.hstack((c, beta1 + beta2))
        c[:, [i]] -= beta2
        x = np.vstack((x, y[i]))
        d += np.outer(c[:, -1], y[i]) + np.outer(-beta2, x[i])

    return c


def krelu_with_sciall(constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float],
                      add_triangle: bool = False, tol=1e-8) -> np.ndarray:
    """
    This is an experimental method.
    This function calculates the convex approximation of the K-ReLU group using the SCI method with all orderings of the
    variables.

    :param constraints: The constraints of the polyhedron. The shape of the constraints should be (m, n+1), where m is
    the number of constraints, and n is the number of variables. The last column of the constraints is the constant
    term.
    :param lower_bounds: The lower bounds of the variables.
    :param upper_bounds: The upper bounds of the variables.
    :param add_triangle: Whether to add the triangle constraints.
    :param tol: The tolerance of the calculation.
    :return: The constraints of the convex approximation of the K-ReLU group.
    """
    assert constraints.shape[1] - 1 == len(lower_bounds) == len(upper_bounds), \
        "The number of variables should be equal to the number of lower bounds and upper bounds."
    assert all(lb < 0 for lb in lower_bounds), "All lower bound should be negative."
    assert all(ub > 0 for ub in upper_bounds), "All upper bound should be positive."

    constraints = constraints.copy()
    vars_num = constraints.shape[1] - 1

    triangle_constraints = None
    if add_triangle or vars_num == 1:
        triangle_constraints = krelu_with_triangle(constraints, lower_bounds, upper_bounds)
    if vars_num == 1:
        return triangle_constraints

    constraints = np.ascontiguousarray(constraints)

    vertices = cal_vertices(constraints)
    vertices = np.ascontiguousarray(np.transpose(vertices))

    if vars_num == 2:
        orders = [[1, 2], [2, 1]]
        inversed_orders = [[1, 2], [2, 1]]
    elif vars_num == 3:
        orders = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
        inversed_orders = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [3, 1, 2], [2, 3, 1], [3, 2, 1]]
    else:
        assert vars_num == 4, "Only support 2 <= k <= 4"
        orders = [[1, 2, 3, 4], [1, 2, 4, 3], [1, 3, 2, 4], [1, 3, 4, 2], [1, 4, 2, 3], [1, 4, 3, 2],
                  [2, 1, 3, 4], [2, 1, 4, 3], [2, 3, 1, 4], [2, 3, 4, 1], [2, 4, 1, 3], [2, 4, 3, 1],
                  [3, 1, 2, 4], [3, 1, 4, 2], [3, 2, 1, 4], [3, 2, 4, 1], [3, 4, 1, 2], [3, 4, 2, 1],
                  [4, 1, 2, 3], [4, 1, 3, 2], [4, 2, 1, 3], [4, 2, 3, 1], [4, 3, 1, 2], [4, 3, 2, 1]]
        inversed_orders = [[1, 2, 3, 4], [1, 2, 4, 3], [1, 3, 2, 4], [1, 4, 2, 3], [1, 3, 4, 2], [1, 4, 3, 2],
                           [2, 1, 3, 4], [2, 1, 4, 3], [3, 1, 2, 4], [4, 1, 2, 3], [3, 1, 4, 2], [4, 1, 3, 2],
                           [2, 3, 1, 4], [2, 4, 1, 3], [3, 2, 1, 4], [4, 2, 1, 3], [3, 4, 1, 2], [4, 3, 1, 2],
                           [2, 3, 4, 1], [2, 4, 3, 1], [3, 2, 4, 1], [4, 2, 3, 1], [3, 4, 2, 1], [4, 3, 2, 1]]

    new_constraints = np.empty((0, 2 * vars_num + 1))
    m = np.matmul(constraints, vertices)

    for k, order in enumerate(orders):
        c = constraints[:, [0] + order].copy()
        x = vertices[[0] + order].copy()
        d = m.copy()
        c = cal_betas(c, x, d)

        inversed_order = inversed_orders[k] + [o + vars_num for o in inversed_orders[k]]
        c[:, 1:] = c[:, inversed_order]
        new_constraints = np.vstack((new_constraints, c))

    if triangle_constraints is not None:
        new_constraints = np.vstack((new_constraints, triangle_constraints))

    return new_constraints
