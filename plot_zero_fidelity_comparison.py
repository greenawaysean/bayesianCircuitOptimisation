import numpy as np
from os import path, getcwd
import pickle
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                   mark_inset)
from matplotlib import cm as cm
from matplotlib import mlab as ml
import matplotlib.lines as mlines
from plotter_params import plot_setup

plot_setup()

qubits = [2, 3, 5]

savefile = path.join(getcwd(), "data", "zero_fidelity_comparison")

p_fidels = []
zero_fidels = []
for nqubits in qubits:
    with open(path.join(savefile, f"{nqubits}_q_process_fidelities.pickle"), 'rb') as f:
        _fs = pickle.load(f)
        p_fidels.append(copy.copy(_fs))

    with open(path.join(savefile, f"{nqubits}_q_zero_fidelities.pickle"), 'rb') as f:
        _fs = pickle.load(f)
        zero_fidels.append(copy.copy(_fs))

fig, ax = plt.subplots(1, 1)
fig.set_figheight(3.3858)

col_cyc = plt.rcParams['axes.prop_cycle'].by_key()['color']

markers = []
labels = []
for i, zero_fidel in enumerate(zero_fidels):
    plt.hexbin(p_fidels[i], zero_fidel, color=col_cyc[i], mincnt=1, bins=1,
               gridsize=300, alpha=0.15)
    markers.append(mlines.Line2D([], [], color=col_cyc[i], marker='.', markersize=10,
                                 lw=0.0, alpha=0.3))
    labels.append(rf"{qubits[i]} qubits")
    plt.plot(p_fidels[i], p_fidels[i], color='black', lw=0.5)

plt.legend(tuple(markers), tuple(labels), frameon=False, handletextpad=0.1, loc=2)


plt.xlabel(r'Process Fidelity (F)')
plt.ylabel(r'$0-$Fidelity ($\tilde{F}_0$)')

axins = inset_axes(ax, width="40%", height="40%", loc=4)
ip = InsetPosition(ax, [0.575, 0.15, 0.425, 0.275])
axins.set_axes_locator(ip)

for i, zero_fidel in enumerate(zero_fidels):
    _log_pf = [np.log10(1-j) for j in p_fidels[i]]
    _log_zf = [np.log10(zero_fidel[j]-k) for j, k in enumerate(p_fidels[i])]
    plt.hexbin(_log_pf, _log_zf, alpha=0.3, gridsize=75, mincnt=1, color=col_cyc[i])
    plt.xlabel(r'$|1-F|$')
    plt.ylabel(r'$|F-\tilde{F}_0|$')
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    plt.yticks([-7, -4, -1], labels=[r'$10^{-7}$', r'$10^{-4}$', r'$10^{-1}$'])
    plt.xticks([-6, -4, -2], labels=[r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$'])
    axins.invert_yaxis()
    axins.invert_xaxis()
    plt.minorticks_off()

plt.savefig(path.join(savefile, "comparing_FOMS_hex.pdf"))
plt.show()
