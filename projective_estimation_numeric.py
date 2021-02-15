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

savepath = path.join(getcwd(), 'data', 'projective_estimations', 'numerics',
                     '3q_random')
savepath = get_filename(savepath)

# number of expectation values and measurement shots - found empirically
num_shots_pf = [4, 8, 16, 64, 144, 256, 400, 576, 784, 1024]
num_circs_pf = [224, 448, 896, 896, 896, 896, 896, 896, 896, 896]
num_shots_zf = [32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
num_circs_zf = [28, 56, 112, 224, 336, 448, 560, 672, 784, 896]

nqubits = 3
num_iters = 100  # 10000
length = 160

params = [2*np.pi*np.random.rand() for i in range(30)]

with open(path.join(savepath, "ideal_params.pickle"), 'wb') as f:
    pickle.dump(params, f)

with open(path.join(savepath, "ideal_params.txt"), 'w') as f:
    for _c in params:
        f.write(str(_c)+'\n')

ideal_U = tensor([qeye(2)]*nqubits)
ideal_U = tensor([generate_u3(params[0], params[1], params[2]),
                  generate_u3(params[3], params[4], params[5]),
                  generate_u3(params[6], params[7], params[8])])*ideal_U
ideal_U = cnot(3, 0, 1)*ideal_U
ideal_U = tensor([generate_u3(params[9], params[10], params[11]),
                  generate_u3(params[12], params[13], params[14]),
                  qeye(2)])*ideal_U
ideal_U = cnot(3, 0, 1)*ideal_U
ideal_U = cnot(3, 1, 2)*ideal_U
ideal_U = tensor([qeye(2),
                  generate_u3(params[15], params[16], params[17]),
                  generate_u3(params[18], params[19], params[20])])*ideal_U
ideal_U = cnot(3, 1, 2)*ideal_U
ideal_U = tensor([generate_u3(params[21], params[22], params[23]),
                  generate_u3(params[24], params[25], params[26]),
                  generate_u3(params[27], params[28], params[29])])*ideal_U

prob_dist = ChiProbDist(nqubits=3, U=ideal_U)
flam_prob_dist = FlammiaProbDist(nqubits=3, U=ideal_U)

r_params = [_c + 0.8*np.random.rand()-0.4 for _c in params]

with open(path.join(savepath, "compare_params.pickle"), 'wb') as f:
    pickle.dump(r_params, f)

with open(path.join(savepath, "compare_params.txt"), 'w') as f:
    for _c in r_params:
        f.write(str(_c)+'\n')

# define list of GateObjs which will be our estimate ansatz
# qiskit and qutip use opposite qubit labelling
ansatz = []
ansatz.append(GateObj('U3', 2, True, (r_params[0], r_params[1], r_params[2])))
ansatz.append(GateObj('U3', 1, True, (r_params[3], r_params[4], r_params[5])))
ansatz.append(GateObj('U3', 0, True, (r_params[6], r_params[7], r_params[8])))
ansatz.append(GateObj('CNOT', [2, 1], False))
ansatz.append(GateObj('U3', 2, True, (r_params[9], r_params[10], r_params[11])))
ansatz.append(GateObj('U3', 1, True, (r_params[12], r_params[13], r_params[14])))
ansatz.append(GateObj('CNOT', [2, 1], False))
ansatz.append(GateObj('CNOT', [1, 0], False))
ansatz.append(GateObj('U3', 1, True, (r_params[15], r_params[16], r_params[17])))
ansatz.append(GateObj('U3', 0, True, (r_params[18], r_params[19], r_params[20])))
ansatz.append(GateObj('CNOT', [1, 0], False))
ansatz.append(GateObj('U3', 2, True, (r_params[21], r_params[22], r_params[23])))
ansatz.append(GateObj('U3', 1, True, (r_params[24], r_params[25], r_params[26])))
ansatz.append(GateObj('U3', 0, True, (r_params[27], r_params[28], r_params[29])))

load = True
if load:
    IBMQ.load_account()
provider = IBMQ.get_provider(
    hub='partner-samsung', group='internal', project='imperial')
backend = Aer.get_backend('qasm_simulator')

qreg = QuantumRegister(3, name='qreg')
init_layout = {0: qreg[0], 1: qreg[1], 6: qreg[2]}

q_ansatz = QiskitAnsatz(ansatz, nqubits, backend, init_layout)
qf_ansatz = QiskitFlammiaAnsatz(ansatz, nqubits, backend, init_layout)

zero_est_circ = QiskitEstimator(prob_dist, q_ansatz, num_shots=1024)
proc_est_circ = QiskitFlammiaEstimator(flam_prob_dist, qf_ansatz, num_shots=1024)

for i in range(len(num_circs_pf)):
    tot_exps = num_circs_pf[i]*num_shots_pf[i]
    pf_circs = num_circs_pf[i]
    zf_circs = num_circs_zf[i]
    proc_est_circ.num_shots = num_shots_pf[i]
    zero_est_circ.num_shots = num_shots_zf[i]

    proc_ests = []
    zero_ests = []
    for j in range(num_iters):
        proc_ests.append(proc_est_circ.estimate_process_fidelity(r_params, pf_circs))
        zero_ests.append(zero_est_circ.estimate_zero_fidelity(r_params, zf_circs))

    with open(path.join(savepath, f"{tot_exps}_experiments_process_fidelity_estimates.pickle"), 'wb') as f:
        pickle.dump(zero_ests, f)

    with open(path.join(savepath, f"{tot_exps}_experiments_process_fidelity_estimates.txt"), 'w') as f:
        for _c in proc_ests:
            f.write(str(_c)+'\n')

    with open(path.join(savepath, f"{tot_exps}_experiments_zero_fidelity_estimates.pickle"), 'wb') as f:
        pickle.dump(zero_ests, f)

    with open(path.join(savepath, f"{tot_exps}_zero_fidelity_estimates.txt"), 'w') as f:
        for _c in zero_ests:
            f.write(str(_c)+'\n')
