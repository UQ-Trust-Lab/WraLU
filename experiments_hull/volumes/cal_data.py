import numpy as np

all_data = {
    2: {3: {}, 4: {}},
    3: {3: {}, 4: {}},
    4: {3: {}, 4: {}},
}

data_dict = {}

for dim in [2, 3, 4]:
    for n in [3, 4]:
        for method in ['cdd', 'triangle', 'fast', 'sci', 'sciplus']:
            file_path = f'./polytopes_{dim}d_{n}_{method}_volume.txt'
            with open(file_path, 'r') as f:
                lines = f.readlines()
            lines = [line.replace('\n', '').split('\t') for line in lines]
            data = [[float(item) for item in line] for line in lines]
            data = np.asarray(data)
            data[:, 3] = data[:, 3] / data[:, 4]
            data = data[:, [0, 2, 3]]
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            all_data[dim][n][method] = [data_mean.tolist(), data_std.tolist()]
            data_dict[(dim, n, method)] = data.copy().tolist()

with open('polytopes_data_dict.txt', 'w') as f:
    f.write(str(data_dict))

for dim in all_data.keys():
    print("Dimension", dim)
    for n in all_data[dim].keys():
        print(f"Constraints number {n}^n")
        times = []
        times_std = []
        constraints_nums = []
        constraints_nums_std = []
        volumes = []
        volumes_std = []
        for method in all_data[dim][n].keys():
            data = all_data[dim][n][method]
            times.append(data[0][0])
            constraints_nums.append(data[0][1])
            volumes.append(data[0][2])
            times_std.append(data[1][0])
            constraints_nums_std.append(data[1][1])
            volumes_std.append(data[1][2])

        print(f"Volume: {dim} & ", end='')
        for v, v_std in zip(volumes, volumes_std):
            print(f"{v:.6f} & {v_std:.6f}", end=' & ')
        print()
        print(f"Time: {dim} & ", end='')
        for t, t_std in zip(times, times_std):
            print(f"{t:.6f} & {t_std:.6f}", end=' & ')
        print()
        print(f"Constraints Number: {dim} & ", end='')
        for c, c_std in zip(constraints_nums, constraints_nums_std):
            print(f"{c:.2f} & {c_std:.2f}", end=' & ')
        print()



