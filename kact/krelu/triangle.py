"""
A function for calculating the triangle relaxation of a k-ReLU group.
"""
from typing import List

import numpy as np


def krelu_with_triangle(constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float]) -> np.ndarray:
    """
    This function calculates the triangle relaxation of a K-ReLU group.

    :param constraints: The constraints of the K-ReLU group. The shape of the constraints should be (m, n+1), where m is
    the number of constraints, and n is the number of variables. The last column of the constraints is the constant
    term.
    :param lower_bounds: The lower bounds of the variables.
    :param upper_bounds: The upper bounds of the variables.
    :return: The triangle relaxation of the K-ReLU group.
    """
    assert constraints.shape[1] - 1 == len(lower_bounds) == len(upper_bounds), \
        "The number of variables in the constraints should be equal to the number of lower bounds and upper bounds."

    # Copy the constraints.
    constraints = constraints.copy()
    vars_num = constraints.shape[1] - 1

    new_constraints = np.empty((0, 2 * vars_num + 1))
    # Add lower constraints.
    # y_i >= 0
    y = np.hstack((np.zeros((vars_num, vars_num + 1)), np.identity(vars_num)))
    new_constraints = np.vstack((new_constraints, y))
    # y_i >= x_i
    yx = np.concatenate((np.zeros((vars_num, 1)), -np.identity(vars_num), np.identity(vars_num)), axis=1)
    new_constraints = np.vstack((new_constraints, yx))

    # Add upper constraints.
    lbs, ubs = np.array([lower_bounds]), np.array([upper_bounds])
    k = ubs / (ubs - lbs)
    b = (- lbs * k).T
    kx = np.diag(k[0])
    y = np.identity(vars_num)
    new_constraints = np.vstack((new_constraints, np.concatenate((b, kx, -y), axis=1)))

    return new_constraints
