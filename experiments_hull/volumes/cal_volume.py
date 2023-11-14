import os
import sys
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
sys.path.insert(0, '../../../')
sys.path.insert(0, '../../../ELINA/')
sys.path.insert(0, '../../../ELINA/python_interface/')
import numpy as np

from kact.volume import VolumeEstimator

if __name__ == '__main__':
    output_constraints_dir = '../output_constraints'
    output_constraints_files = os.listdir(output_constraints_dir)
    output_constraints_files = [file for file in output_constraints_files if file.endswith('.txt')]
    output_constraints_files.sort(key=lambda x: int(x.split('.')[-2].split('_')[1][:-1]))

    for output_constraints_file in output_constraints_files:
        args = output_constraints_file.split('.')[0].split('_')
        dim = int(args[1].replace('d', ''))
        method = args[-1]
        if dim != 4:
            continue
        if method not in ['cdd']:
            continue

        print(f'[INFO] Processing {output_constraints_file}...')
        e = int(args[2])
        method = args[3]

        max_points_dict = {
            2: 1000,
            3: 100,
            4: 10,
        }
        max_points_in = max_points_dict[dim]

        with open(os.path.join(output_constraints_dir, output_constraints_file), 'r') as f:
            lines = f.readlines()

        saved_file_path = output_constraints_file.replace('.txt', '_volume.txt')
        file = open(saved_file_path, 'w')
        for i, line in enumerate(lines):
            line.replace('\n', '')
            line = line.split('\t')
            time_cal = float(line[0])
            if method in ['sci', 'sciplus']:
                time_vertices = float(line[1])
            else:
                time_vertices = 0

            constraints_num = eval(line[-4])[0]
            output_constraints = np.asarray(eval(line[-3]))
            lb = eval(line[-2])
            ub = eval(line[-1])

            volume_estimator = VolumeEstimator(output_constraints, lb, ub)
            points_in_num, points_num = volume_estimator.estimate('random',
                                                                  lower_bounds=lb, upper_bounds=ub,
                                                                  sample_area_shape='box',
                                                                  max_points_in=max_points_in)
            volume = points_in_num  # / points_num
            print(f'{i}/{len(lines)}-{points_in_num}/{points_num}={points_in_num/points_num:.6f}', flush=True)
            file.write(f'{time_cal}\t{time_vertices}\t{constraints_num}\t{volume}\t{points_num}\n')
        print()
        file.close()

