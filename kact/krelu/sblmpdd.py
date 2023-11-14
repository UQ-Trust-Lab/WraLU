"""
This module implements the SBLM-PDD algorithm in PRIMA for computing the convex approximation of a k-ReLU group.
"""
import warnings
from typing import List

import numpy as np
try:
    import sys
    sys.path.insert(0, '../../')
    from ELINA.python_interface import fconv
except ImportError:
    # warnings.warn("Please install ELINA first. SBLM-PDD for relu is not available.")
    fkrelu = None


def fkrelu(constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float]) -> np.ndarray:
    """
    This function calculates the convex approximation of a k-ReLU group by the SBLM-PDD algorithm in PRIMA.

    :param constraints: The constraints of the K-ReLU group. The shape of the constraints should be (m, n+1), where m is
    the number of constraints, and n is the number of variables. The last column of the constraints is the constant
    term.
    :param lower_bounds: The lower bounds of the variables.
    :param upper_bounds: The upper bounds of the variables.
    :return: The convex approximation of the K-ReLU group.
    """
    assert constraints.shape[1] - 1 == len(lower_bounds) == len(upper_bounds), \
        "constraints, lower_bounds and the upper_bounds should have the same number of variables."
    assert all(lb < 0 for lb in lower_bounds), "All lower bound should be negative."
    assert all(ub > 0 for ub in upper_bounds), "All upper bound should be positive."
    return fconv.fkrelu(constraints)
