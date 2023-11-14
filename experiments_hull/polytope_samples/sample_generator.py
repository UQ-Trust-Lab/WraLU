import itertools

import numpy as np
from gurobipy import GRB

from kact.utils import solve_lp
from kact.volume import InputConstraintsGenerator, VolumeCalculator


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


def generate_samples(dim: int, num: int, saved_file_path: str):
    input_constrs_generator = InputConstraintsGenerator(dim)
    input_constrs_method = 'box+random'
    input_constrs_box_lower_bound = -5
    input_constrs_box_upper_bound = 5
    for n in [3, 4]:
        print(f'Generating {dim}d polytope with {dim}^{n} constriants...')
        saved_file = f'{saved_file_path[:-4]}_{n}.txt'
        input_constrs_num = dim ** n
        file = open(saved_file, 'w')
        file2 = None
        if dim <= 4:
            saved_file2 = f'{saved_file_path[:-4]}_{n}_oct.txt'
            file2 = open(saved_file2, 'w')

        for _ in range(num):
            input_constrs = input_constrs_generator.generate(input_constrs_method,
                                                             input_constrs_box_lower_bound,
                                                             input_constrs_box_upper_bound,
                                                             input_constrs_num)

            file.write(str(input_constrs.tolist()) + '\n')
            if file2 is not None:
                input_constrs_oct = _create_octahedron_approximation(input_constrs)
                file2.write(str(input_constrs_oct.tolist()) + '\n')

        file.close()
        if file2 is not None:
            file2.close()


if __name__ == '__main__':

    num = 30
    for dim in range(2, 9):
        saved_file_path = f'./polytopes_{dim}d.txt'
        generate_samples(dim, num, saved_file_path)
        print(f'Generated {dim}d polytopes with {num} samples.')
