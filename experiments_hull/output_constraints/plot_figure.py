import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

all_data_mean = {'sci': {3: [], 4: []}, 'sciplus': {3: [], 4: []}}
# all_data_std = {'sci': {3: [], 4: []}, 'sciplus': {3: [], 4: []}}
dimensions = [2, 3, 4, 5, 6, 7, 8]

print('[INFO] Loading data...')
for method in ['sci', 'sciplus']:
    for d in dimensions:
        for n in [3, 4]:
            file_path = f'./polytopes_{d}d_{n}_{method}.txt'
            file = open(file_path, 'r')
            lines = file.readlines()
            file.close()
            lines = [line.split('\t')[:2] for line in lines]
            lines = [[float(num) for num in line] for line in lines]
            data = np.asarray(lines)
            data_mean = np.mean(data, axis=0)
            all_data_mean[method][n].append(data_mean)
            # data_std = np.std(data, axis=0)
            # all_data_std[method][n].append(data_std)
print('[INFO] Data loaded.')

print('[INFO] Plotting...')
# plt.rc('text', usetex=True)
# Set font to Times New Roman
plt.rc('font', family='Times New Roman')
fig = plt.figure(figsize=(14, 5))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0)  # Set the margins
plt.tight_layout()

indices = ['a', 'b']
methods = ['sci', 'sciplus']
for i in range(2):
    method = methods[i]
    ax = plt.subplot(1, 2, i + 1)
    ax2 = ax.twinx()
    ax, ax2 = ax2, ax
    ax.set_xlabel('Input Dimension', fontsize=14)
    ax.set_ylabel('Time(s)', fontsize=14)
    ax.set_yscale('log')
    ax2.set_ylabel('Ratio of Vertices Calculation', fontsize=14)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(dimensions)
    ax.set_title(f'({indices[i]}) Average Runtime of {method.upper()}', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    for n in [3, 4]:
        data = np.asarray(all_data_mean[method][n])
        total_time = data[:, 0].reshape(-1)
        part_time = data[:, 1].reshape(-1)

        for j in range(len(total_time)):
            t = total_time[j]
            p = part_time[j]

            # if n == 3:
            #     if j == 0:
            #         ax.bar(dimensions[j] - 0.2, p, color='white', width=0.4, alpha=0.5, edgecolor='black', hatch='//',
            #                label='Vertices Calculation ($3^n$ constraints)')
            #         ax.bar(dimensions[j] - 0.2, t - p, bottom=p, color='#808080', width=0.4, alpha=0.5, edgecolor='black', hatch='//',
            #                label='Other ($3^n$ constraints)')
            #
            #     else:
            #         ax.bar(dimensions[j] - 0.2, p, color='white', width=0.4, alpha=0.5, edgecolor='black', hatch='//')
            #         ax.bar(dimensions[j] - 0.2, t - p, bottom=p, color='#808080', width=0.4, alpha=0.5, edgecolor='black', hatch='//')
            # else:
            #     if j == 0:
            #         ax.bar(dimensions[j] + 0.2, p, color='white', width=0.4, alpha=0.5, edgecolor='black', hatch='---',
            #                label='Vertices Calculation ($4^n$ constraints)')
            #         ax.bar(dimensions[j] + 0.2, t - p, bottom=p, color='#808080', width=0.4, alpha=0.5, edgecolor='black', hatch='---',
            #                label='Other ($4^n$ constraints)')
            #     else:
            #         ax.bar(dimensions[j] + 0.2, p, color='white', width=0.4, alpha=0.5, edgecolor='black', hatch='---')
            #         ax.bar(dimensions[j] + 0.2, t - p, bottom=p, color='#808080', width=0.4, alpha=0.5, edgecolor='black', hatch='---')

            if n == 3:
                if j == 0:
                    bar1 = ax2.bar(dimensions[j] - 0.2, p/t, color='white', width=0.4, alpha=0.5, edgecolor='black', hatch='//',
                           label='Ratio of Vertices Calculation ($3^n$ constraints)')
                else:
                    ax2.bar(dimensions[j] - 0.2, p/t, color='white', width=0.4, alpha=0.5, edgecolor='black', hatch='//')
                line1 = ax.plot(dimensions, total_time, 'ko-', label='Total Runtime ($3^n$ constraints)', linewidth=0.5)
            else:
                if j == 0:
                    bar2 = ax2.bar(dimensions[j] + 0.2, p/t, color='white', width=0.4, alpha=0.5, edgecolor='black', hatch='---',
                           label='Ratio of Vertices Calculation ($4^n$ constraints)')
                else:
                    ax2.bar(dimensions[j] + 0.2, p/t, color='white', width=0.4, alpha=0.5, edgecolor='black', hatch='---')
                line2 = ax.plot(dimensions, total_time, 'k*-', label='Total Runtime ($4^n$ constraints)', linewidth=0.5)

    # ax.legend(loc='center left', ncol=1)
    # ax2.legend(loc='upper left', ncol=1)
    # Combine the labels of ax and ax2
    obj = line1 + line2 + [bar1, bar2]
    labs = [l.get_label() for l in obj]
    ax.legend(obj, labs, loc='upper left', ncol=1)


fig.show()

# Save the figure
fig.savefig('alg_runtime_ratio.png', dpi=300)
