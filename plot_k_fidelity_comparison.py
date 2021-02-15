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

nqubits = 3
orders = range(nqubits+1)
savefile = path.join(getcwd(), "data", "k_fidelity_comparison", f"{nqubits}_qubits")

k_fidels = []
p_fidels = []
for k in orders:
    with open(path.join(savefile, f"{k}_fidelities.pickle"), 'rb') as f:
        _fs = pickle.load(f)
        k_fidels.append(copy.copy(_fs[:]))

    with open(path.join(savefile, f"{k}_process_fidelities.pickle"), 'rb') as f:
        _pfs = pickle.load(f)[:]
        p_fidels.append(_pfs)

# main plot
fig, ax = plt.subplots(1, 1)
fig.set_figheight(3.3858)

col_cyc = plt.rcParams['axes.prop_cycle'].by_key()['color']

markers = []
labels = []
for i, k_fidel in enumerate(k_fidels):
    plt.hexbin(p_fidels[i], k_fidel, color=col_cyc[i], mincnt=1, bins=5, gridsize=300,
               alpha=0.15)
    markers.append(mlines.Line2D([], [], color=col_cyc[i], marker='.', markersize=10,
                                 lw=0.0, alpha=0.3))
    labels.append(rf'k={orders[k]}')

plt.legend(tuple(markers), tuple(labels), frameon=False, handletextpad=0.1, loc=2)

plt.xlabel(r'Process Fidelity (F)')
plt.ylabel(r'$k-$Fidelity ($\tilde{F}_k$)')

# log plot inset
axins = inset_axes(ax, width="40%", height="40%", loc=4)
ip = InsetPosition(ax, [0.575, 0.15, 0.425, 0.275])
axins.set_axes_locator(ip)
for i, k_fidel in enumerate(k_fidels):
    # the n-fidelity is exactly the process fidelity, so plotting it is only plotting
    # numerical error around 1e-15 which distorts the rest of the plot and thus we
    # don't plot it.
    if i == nqubits:
        continue
    _log_pf = [np.log10(1-j) for j in p_fidels[i]]
    _log_kf = [np.log10(k_fidel[j]-k) for j, k in enumerate(p_fidels[i])]
    plt.hexbin(_log_pf, _log_kf, alpha=0.3, gridsize=75, mincnt=1, color=col_cyc[i])

    plt.xlabel(r'$|1-F|$')
    plt.ylabel(r'$|F-\tilde{F}_k|$')
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    plt.yticks([-7, -4, -1], labels=[r'$10^{-7}$', r'$10^{-4}$', r'$10^{-1}$'])
    plt.xticks([-6, -4, -2], labels=[r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$'])
    axins.invert_yaxis()
    axins.invert_xaxis()
    plt.minorticks_off()

plt.savefig(path.join(savefile, "n3_all_orders_inset_diff_hex.pdf"))
plt.show()
