import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from utils import generate_u3
from estimators import QiskitAnsatz, QiskitFlammiaAnsatz, QiskitEstimator, QiskitFlammiaEstimator
from probability_distributions import ChiProbDist, FlammiaProbDist
from utils import GateObj, U_from_hamiltonian, get_filename
from qutip import Qobj, cnot, tensor, qeye
from qiskit.aqua import QuantumInstance
from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
import pickle
from os import path, getcwd, makedirs
import sys
import matplotlib.pyplot as plt
zero_fidelsimport numpy as np


savepath = path.join(getcwd(), 'data', 'projective_estimations', 'real_machine',
                     '3q_random')


with open(path.join(savepath, 'process_fidelity_estimates.pickle'), 'rb') as f:
    proc_fidels = pickle.load(f)

with open(path.join(savepath, 'zero_fidelity_estimates.pickle'), 'rb') as f:
    zero_fidesl = pickle.load(f)

point_alpha = 0.8
fill_alpha = 0.12

x = np.linspace(0, 1, 11)
y = np.linspace(0, 1, 11)

plt.scatter([i for i in range(len(proc_fidels))], [
            f for f in proc_fidels], marker='x', s=15, alpha=point_alpha)
plt.fill_between([-1 + i for i in range(len(proc_fidels)+2)], f_mean-np.std(proc_fidels),
                 f_mean+np.std(proc_fidels), alpha=fill_alpha, color=col_cyc[0])
plt.hlines(np.mean(proc_fidels), -1, len(proc_fidels), lw=0.7, color=col_cyc[0])
p1 = mlines.Line2D([], [], color=col_cyc[0], marker='x', markersize=np.sqrt(15), lw=0.7)
p2 = mpatches.Patch(color=col_cyc[0], alpha=fill_alpha, linewidth=0)

plt.scatter([i for i in range(len(zero_fidels))], [
            f for f in zero_fidels], marker='^', s=15, alpha=point_alpha)
plt.fill_between([-1 + i for i in range(len(zero_fidels)+2)], fom_mean-np.std(zero_fidels),
                 fom_mean+np.std(zero_fidels), alpha=fill_alpha, color=col_cyc[1])
plt.hlines(np.mean(zero_fidels), -1, len(zero_fidels), lw=0.7, color=col_cyc[1])
p3 = mlines.Line2D([], [], color=col_cyc[1], marker='^', markersize=np.sqrt(15), lw=0.7)
p4 = mpatches.Patch(color=col_cyc[1], alpha=fill_alpha, linewidth=0)

plt.xlabel("Evaluation")
plt.ylabel("Estimated value")

plt.legend(((p1, p2), (p3, p4),), ('Process Fidelity', r"$0-$Fidelity",), frameon=True)
plt.savefig(path.join(savepath, "toronto_50evals_129024_exps_mean_stds.png"))
