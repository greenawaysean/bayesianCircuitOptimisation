import numpy as np
import sys
import copy
from scipy import linalg
from os import path, getcwd, makedirs
import pickle
from qutip import Qobj
from utils import get_filename, generate_rand_herm
from probability_distributions import ChiProbDist, FlammiaProbDist
from estimators import QutipAnsatz, QutipEstimator, QutipFlammiaEstimator
from fidelity_functions import process_fidelity, zero_fidelity

nqubits = 3
epsilon = 0.1
num_iters = 10000
length = 160
dims = [[2]*nqubits, [2]*nqubits]

savefile = path.join(getcwd(), "data", "full_trace_estimations",
                     f"epsilon_{epsilon}_length_{length}".replace('.', '_'))
savefile = get_filename(savefile)

H, ideal_coeffs = generate_rand_herm(nqubits)
U = linalg.expm(-1j*H)

with open(path.join(savefile, "ideal_coeffs.pickle"), 'wb') as f:
    pickle.dump(ideal_coeffs, f)

with open(path.join(savefile, "ideal_coeffs.txt"), 'w') as f:
    for _c in ideal_coeffs:
        f.write(str(_c)+'\n')

_h, rot_coeffs = generate_rand_herm(nqubits)
Vrot = linalg.expm(-1j*epsilon*_h)
Vrot_dag = np.conj(np.transpose(Vrot))
V = np.dot(np.dot(Vrot, U), Vrot_dag)

with open(path.join(savefile, "rot_coeffs.pickle"), 'wb') as f:
    pickle.dump(rot_coeffs, f)

with open(path.join(savefile, "rot_coeffs.txt"), 'w') as f:
    for _c in rot_coeffs:
        f.write(str(_c)+'\n')

true_zf = zero_fidelity(copy.copy(U), copy.copy(V), nqubits)
true_pf = process_fidelity(copy.copy(U), copy.copy(V), nqubits)

with open(path.join(savefile, "true_zero_fidelity.pickle"), 'wb') as f:
    pickle.dump(true_zf, f)

with open(path.join(savefile, "true_zero_fidelity.txt"), 'w') as f:
    f.write(str(true_zf)+'\n')

with open(path.join(savefile, "true_process_fidelity.pickle"), 'wb') as f:
    pickle.dump(true_pf, f)

with open(path.join(savefile, "true_process_fidelity.txt"), 'w') as f:
    f.write(str(true_pf)+'\n')

q_ansatz = QutipAnsatz(nqubits, V=Qobj(V, dims=dims))

zero_prob_dist = ChiProbDist(nqubits, Qobj(U, dims=dims))
proc_prob_dist = FlammiaProbDist(nqubits, Qobj(U, dims=dims))

zero_estimator = QutipEstimator(zero_prob_dist, nqubits, q_ansatz)
proc_estimator = QutipFlammiaEstimator(proc_prob_dist, nqubits, q_ansatz)

zero_ests = []
proc_ests = []
for i in range(num_iters):
    zero_ests.append(zero_estimator.calculate_zf(length=length))
    proc_ests.append(proc_estimator.calculate_pf(length=length))

with open(path.join(savefile, "zero_fidelity_estimates.pickle"), 'wb') as f:
    pickle.dump(zero_ests, f)

with open(path.join(savefile, "zero_fidelity_estimates.txt"), 'w') as f:
    for _c in zero_ests:
        f.write(str(_c)+'\n')

with open(path.join(savefile, "process_fidelity_estimates.pickle"), 'wb') as f:
    pickle.dump(proc_ests, f)

with open(path.join(savefile, "process_fidelity_estimates.txt"), 'w') as f:
    for _c in proc_ests:
        f.write(str(_c)+'\n')
