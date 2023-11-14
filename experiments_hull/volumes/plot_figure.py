import numpy as np

with open('polytopes_data_dict.txt', 'r') as f:
    data_dict = eval(f.read())

for k, v in data_dict.items():
    data_dict[k] = np.array(v)

data_mean = {'cdd': [], 'triangle': [], 'fast': [], 'sci': [], 'sciplus': []}
data_std = {'cdd': [], 'triangle': [], 'fast': [], 'sci': [], 'sciplus': []}
for k, v in data_dict.items():
    method = k[2]
    v = v[:, [2, 0, 1]]
    data_mean[method].append(np.mean(v, axis=0).tolist())
    data_std[method].append(np.std(v, axis=0).tolist())

print(data_mean)
print(data_std)
# Plotting
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 10))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.25)  # Set the margins
plt.tight_layout()
title = ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)']
y_labels = ['Relative Volume', 'Runtime(s)', 'Number of Constraints']
for i in range(2):
    for j in range(1, 4):
        ax = fig.add_subplot(2, 3, 3 * i + j)
        ax.set_title(f"{title[3 * i + j - 1]} {y_labels[j - 1]} (${i+3}^n$ Constraints)")
        ax.set_xlabel('Dimension', fontsize=12)
        # ax.set_ylabel(y_labels[j - 1], fontsize=12)
        ax.set_xticks([2, 3, 4])
        # ax.set_yticks(fontsize=12)
        ax.set_yscale('log')

        # Plot a error bar
        x = np.arange(2, 5)
        y_cdd = np.array(data_mean['cdd'][3 * i: 3 * i + 3])[:, j - 1]

        y_triangle = np.array(data_mean['triangle'][3 * i: 3 * i + 3])[:, j - 1]
        y_fast = np.array(data_mean['fast'][3 * i: 3 * i + 3])[:, j - 1]
        y_sci = np.array(data_mean['sci'][3 * i: 3 * i + 3])[:, j - 1]
        y_sciplus = np.array(data_mean['sciplus'][3 * i: 3 * i + 3])[:, j - 1]

        yerr_cdd = np.array(data_std['cdd'][3 * i: 3 * i + 3])[:, j - 1]
        yerr_triangle = np.array(data_std['triangle'][3 * i: 3 * i + 3])[:, j - 1]
        yerr_fast = np.array(data_std['fast'][3 * i: 3 * i + 3])[:, j - 1]
        yerr_sci = np.array(data_std['sci'][3 * i: 3 * i + 3])[:, j - 1]
        yerr_sciplus = np.array(data_std['sciplus'][3 * i: 3 * i + 3])[:, j - 1]

        # Plot the mean line
        ax.plot(x, y_cdd, 'o-', label='cdd')
        ax.plot(x, y_triangle, 'o-', label='triangle')
        ax.plot(x, y_fast, 'o-', label='fast')
        ax.plot(x, y_sci, 'o-', label='sci')
        ax.plot(x, y_sciplus, 'o-', label='sciplus')

        # Plot the between line
        ax.fill_between(x, y_cdd - yerr_cdd, y_cdd + yerr_cdd, alpha=0.2)
        ax.fill_between(x, y_triangle - yerr_triangle, y_triangle + yerr_triangle, alpha=0.2)
        ax.fill_between(x, y_fast - yerr_fast, y_fast + yerr_fast, alpha=0.2)
        ax.fill_between(x, y_sci - yerr_sci, y_sci + yerr_sci, alpha=0.2)
        ax.fill_between(x, y_sciplus - yerr_sciplus, y_sciplus + yerr_sciplus, alpha=0.2)

        ax.legend(fontsize=12)
plt.show()

# Save the figure
fig.savefig('alg_comparison.png', dpi=300, bbox_inches='tight')