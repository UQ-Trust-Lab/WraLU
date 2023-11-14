import os

import numpy as np

from kact.utils import get_bounds_of_variables


def cal_bounds(polytope_samples_file_path: str):

    with open(polytope_samples_file_path, 'r') as f:
        lines = f.readlines()
    polytope_samples_file_path = ('./' + polytope_samples_file_path.split('\\')[-1]).replace('.txt', '_bounds.txt')
    dim = int(polytope_samples_file_path.split('_')[1][:-1])

    print(f'[INFO] Processing {polytope_samples_file_path}...')
    output_file = open(polytope_samples_file_path, 'w')

    for line in lines:
        constraints = np.asarray(eval(line))
        lower_bounds, upper_bounds = get_bounds_of_variables(constraints)
        output_file.write(f'({lower_bounds}, {upper_bounds})\n')

    output_file.close()


if __name__ == '__main__':
    # Read all txt files in ../polytope_samples
    polytope_samples_folder = '../polytope_samples'
    polytope_samples_files = os.listdir(polytope_samples_folder)
    polytope_samples_files = [file for file in polytope_samples_files if
                              file.endswith('.txt') and not file.endswith('oct.txt')]
    polytope_samples_files.sort(key=lambda x: int(x.split('.')[-2].split('_')[1][:-1]))

    for polytope_samples_file in polytope_samples_files:
        print(f'[INFO] Processing {polytope_samples_file}...')
        polytope_samples_file_path = os.path.join(polytope_samples_folder, polytope_samples_file)
        cal_bounds(polytope_samples_file_path)
