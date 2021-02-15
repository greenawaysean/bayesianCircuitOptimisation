import numpy as np
import sys
from os import path, getcwd, makedirs
import pickle
from utils import get_filename
from plotter_params import plot_setup
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.ticker as ticker

plot_setup()

epsilon = 0.1
length = 160

filename = path.join(getcwd(), "data", "full_trace_estimations",
                     f"epsilon_{epsilon}_length_{length}".replace('.', '_'))


with open(path.join(filename, 'true_process_fidelity.pickle'), 'rb') as f:
    true_pf = pickle.load(f)

with open(path.join(filename, 'true_zero_fidelity.pickle'), 'rb') as f:
    true_zf = pickle.load(f)

with open(path.join(filename, 'process_fidelity_estimates.pickle'), 'rb') as f:
    proc_fs = pickle.load(f)

with open(path.join(filename, 'zero_fidelity_estimates.pickle'), 'rb') as f:
    zero_fs = pickle.load(f)


nbin = 80

pf_diffs = [i - true_pf for i in proc_fs[::]]
pf_range = np.max(pf_diffs) - np.min(pf_diffs)
pf_width = pf_range/nbin


p_fidels = sorted(pf_diffs)
fmean = np.mean(p_fidels)
fx, fy, _ = plt.hist(p_fidels, bins=nbin, alpha=0.6, density=True, label='Fidelity')
f_fit = stats.norm.pdf(p_fidels, fmean, np.std(p_fidels))
f_fit = f_fit*(np.max(fx)/np.max(f_fit))
plt.plot(p_fidels, f_fit, marker=None, lw=0.75)
plt.xlabel("Approximation error")
plt.ylabel("Frequency")
plt.xlim(-0.5, 0.5)


zf_diffs = [i - true_zf for i in zero_fs[::]]
zf_range = np.max(zf_diffs) - np.min(zf_diffs)
zf_nbins = np.int(zf_range/pf_width)

z_fidels = sorted(zf_diffs)
z_mean = np.mean(z_fidels)
zfx, zfy, _ = plt.hist(z_fidels, bins=zf_nbins, alpha=0.6, density=True, label='Figure of Merit')
z_fit = stats.norm.pdf(z_fidels, z_mean, np.std(z_fidels))
z_fit = z_fit*(np.max(zfx)/np.max(z_fit))
plt.plot(z_fidels, z_fit, marker=None, lw=0.75)

plt.xlabel("Approximation error")
plt.ylabel("Frequency")

plt.tight_layout(w_pad=-1)
plt.legend()
plt.savefig(path.join(filename, 'F_v_FOM_dists_160_evals_normed_eq_width.pdf'))
plt.show()
