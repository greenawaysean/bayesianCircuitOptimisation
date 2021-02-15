import numpy as np
import pickle
from os import path, getcwd, makedirs

import matplotlib.pyplot as plt
from plotter_params import plot_setup
from matplotlib.ticker import MaxNLocator

filename = path.join('data', 'BO', 'real_machine')
reg_Fs = []
exp_Fs = []
for i in range(1, 7):
    if i == 1:
        filepath = path.join(filename, '3q_CNOT_three_applications')
    else:
        filepath = path.join(filename, f'3q_CNOT_three_applications_{i}')
    with open(path.join(filepath, 'fidels.pickle'), 'rb') as f:
        regular_true_F, seen_true_F, exp_opt_true_F = pickle.load(f)
        print(regular_true_F)
    reg_Fs.append(regular_true_F)
    exp_Fs.append(exp_opt_true_F)

if not path.exists(path.join(filename, 'plots')):
    makedirs(path.join(filename, 'plots'))

_s = 10
plot_setup()
plt.scatter([i+1 for i in range(len(reg_Fs))], reg_Fs, marker='o', label='Unoptimised', s=_s)
plt.scatter([i+1 for i in range(len(exp_Fs))], exp_Fs, marker='^', label='Optimised', s=_s)
plt.ylim([0., 0.8])
plt.xlabel('Optimisation Run')
plt.ylabel('Process Fidelity')
plt.legend(frameon=True)
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.margins(0.01)
plt.savefig(path.join(filename, 'plots', '70_iterations_meas_err_simplified.pdf'))
plt.show()
