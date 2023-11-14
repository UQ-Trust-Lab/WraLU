"""
This module provides the function to solve linear programming problem by GUROBI.
"""
import itertools
from typing import Optional, Tuple, List

import numpy as np
import gurobipy as grb
from gurobipy import GRB


def solve_lp(constraints: np.ndarray, obj_func: np.ndarray, obj_type: GRB) -> Optional[Tuple[List, float]]:
    """
    Solve linear programming problem by GUROBI.

    :param constraints: The constraints of the linear programming problem.
    :param obj_func: The objective function of the linear programming problem.
    :param obj_type: The type of the objective function, GRB.MAXIMIZE or GRB.MINIMIZE.
    :return: The optimal solution and the optimal value of the linear programming problem.
    """
    model = grb.Model("Solve LP by GUROBI")
    model.setParam("OutputFlag", False)
    model.setParam("LogToConsole", 0)
    model.setParam("Method", 0)  # Simplex method
    vars_num = constraints.shape[1] - 1
    # The default ub is GRB.INFINITY and the default lb is 0, here change the lb.
    x = np.asarray([1] + [model.addVar(lb=-GRB.INFINITY) for _ in range(vars_num)]).reshape((vars_num + 1, 1))

    for constraint in constraints:
        model.addConstr(grb.LinExpr(np.dot(constraint, x)[0]) >= 0)

    model.setObjective(grb.LinExpr(np.dot(obj_func, x)[0]), obj_type)
    model.optimize()
    return (model.x, model.objVal) if model.status == GRB.OPTIMAL else None


def get_bounds_of_variables(constraints: np.ndarray) -> Tuple[List, List]:
    upper_bounds, lower_bounds = [], []
    vars_num = constraints.shape[1] - 1
    for i in range(1, vars_num + 1):
        obj_func = np.zeros((1, vars_num + 1))
        obj_func[0, i] = 1
        _, upper_bound = solve_lp(constraints, obj_func[0], GRB.MAXIMIZE)
        _, lower_bound = solve_lp(constraints, obj_func[0], GRB.MINIMIZE)
        assert upper_bound is not None and lower_bound is not None, \
            "The polytope is unbounded or the LPP is infeasible."
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    return lower_bounds, upper_bounds


def get_octahedral_approximation(constraints: np.ndarray) -> np.ndarray:
    oct_constraints = []
    dim = constraints.shape[1] - 1
    for coeffs in itertools.product([-1, 0, 1], repeat=dim):
        coeffs = list(coeffs)
        if all(c == 0 for c in coeffs):
            continue
        obj_func = [0] + coeffs
        _, upper_bound = solve_lp(constraints, np.asarray(obj_func), GRB.MAXIMIZE)
        oct_constraints.append([upper_bound] + [-c for c in coeffs])

    return np.asarray(oct_constraints)
