import numpy as np
import matplotlib.pyplot as plt
import sys
from os import path, getcwd, makedirs
import pickle
from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qutip import Qobj, cnot, tensor, qeye
from utils import GateObj, U_from_hamiltonian, get_filename
from probability_distributions import ChiProbDist, FlammiaProbDist
from estimators import QiskitAnsatz, QiskitFlammiaAnsatz, QiskitEstimator, QiskitFlammiaEstimator
from utils import generate_u3
from plotter_params import plot_setup

plot_setup()

filename = path.join(getcwd(), 'data', 'projective_estimations', 'numerics',
                     '3q_random')

num_shots_pf = [4, 8, 16, 64, 144, 256, 400, 576, 784, 1024]
num_circs_pf = [224, 448, 896, 896, 896, 896, 896, 896, 896, 896]
num_exps = [shot*num_circs_pf[i] for i, shot in enumerate(num_shots_pf)]

proc_stds = []
zero_stds = []
for exp in num_exps:
    with open(path.join(filename, f"{exp}_experiments_process_fidelity_estimates.pickle"), 'rb') as f:
        proc_ests = pickle.load(f)

    with open(path.join(filename, f"{exp}_experiments_zero_fidelity_estimates.pickle"), 'rb') as f:
        zero_ests = pickle.load(f)

    proc_stds.append(np.std(proc_ests))
    zero_stds.append(np.std(zero_ests))

plt.scatter(num_exps, proc_stds, label="Process Fidelity", marker='x', s=15)
plt.scatter(num_exps, zero_stds, label="$0-$Fidelity", marker='x', s=15)
plt.legend(frameon=True, loc='upper right')
ax = plt.gca()
ax.set_xscale('log')
plt.xlabel("Total experiments")
plt.ylabel("Standard deviation")
ax.margins(0.02)

plt.savefig(path.join(filename, "std_progresion.pdf"))
plt.show()
