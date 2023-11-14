import os
import time
import sys

cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
sys.path.insert(0, '../../../')
sys.path.insert(0, '../../../ELINA/')
sys.path.insert(0, '../../../ELINA/python_interface/')

import numpy as np
from kact import fkrelu, krelu_with_pycdd, krelu_with_sci, krelu_with_sciplus, krelu_with_triangle, krelu_with_sci_redundant

def read_constraints_and_bounds(constraints_file_path: str, bounds_file_path: str):
    with open(constraints_file_path, 'r') as f:
        constraints = f.readlines()
    with open(bounds_file_path, 'r') as f:
        bounds = f.readlines()
    constraints = [eval(constraint) for constraint in constraints]
    bounds = [eval(bound) for bound in bounds]
    return constraints, bounds


if __name__ == '__main__':
    constraints_dir = '../polytope_samples'
    bounds_dir = '../polytope_bounds'
    constraints_files = os.listdir(constraints_dir)
    constraints_files = [file for file in constraints_files if file.endswith('.txt')]
    # Sort by dimension
    constraints_files.sort(key=lambda x: int(x.split('.')[-2].split('_')[1][:-1]))

    for method in ['triangle', 'fast', 'sci', 'sciplus', 'cdd']:
        for i, constraints_file in enumerate(constraints_files):
            if method == 'fast' and constraints_file.endswith('oct.txt'):
                dim = int(constraints_file.split('.')[-2].split('_')[-3][:-1])
                constraints_file_path = os.path.join(constraints_dir, constraints_file)
                bounds_file_path = os.path.join(bounds_dir, constraints_file.replace('_oct.txt', '_bounds.txt'))
            elif not constraints_file.endswith('oct.txt') and method != 'fast':
                dim = int(constraints_file.split('.')[-2].split('_')[-2][:-1])
                constraints_file_path = os.path.join(constraints_dir, constraints_file)
                bounds_file_path = os.path.join(bounds_dir, constraints_file.replace('.txt', '_bounds.txt'))
            else:
                continue

            print(f'[INFO] Processing {constraints_file_path} with {method}...')

            constraints_list, bounds_list = read_constraints_and_bounds(constraints_file_path, bounds_file_path)

            bounds_file_path = os.path.basename(bounds_file_path)
            saved_file_path = bounds_file_path.replace('_bounds.txt', f'_{method}.txt').split('\\')[-1]

            file = open(saved_file_path, 'w')
            for n, (constraints, bounds) in enumerate(zip(constraints_list, bounds_list)):
                print(f'[INFO] Processing {n + 1}/{len(constraints_list)} with {method}...', end='')
                lb, ub = bounds

                constraints = np.asarray(constraints)
                time_cal = time.time()

                time_v = None
                if method == 'cdd':
                    output_constraints = krelu_with_pycdd(constraints, lb, ub)
                elif method == 'fast':
                    output_constraints = fkrelu(constraints, lb, ub)
                elif method == 'sci':
                    output_constraints, time_v = krelu_with_sci(constraints, lb, ub, add_triangle=True,
                                                                output_time_cal_vertices=True)
                elif method == 'sciplus':
                    output_constraints, time_v = krelu_with_sciplus(constraints, lb, ub, add_triangle=True,
                                                                    output_time_cal_vertices=True)
                elif method == 'sciexp':
                    output_constraints = krelu_with_sci_redundant(constraints, lb, ub, add_triangle=True)
                elif method == 'triangle':
                    output_constraints = krelu_with_triangle(constraints, lb, ub)
                else:
                    raise NotImplementedError

                time_cal = time.time() - time_cal

                if time_v is None:
                    file.write(f'{time_cal:.10f}\t'
                               f'{output_constraints.shape}\t{output_constraints.tolist()}\t{lb}\t{ub}\n')
                else:
                    file.write(f'{time_cal:.10f}\t{time_v:.10f}\t'
                               f'{output_constraints.shape}\t{output_constraints.tolist()}\t{lb}\t{ub}\n')
                print(f' Time: {time_cal:.10f} s')
            file.close()
