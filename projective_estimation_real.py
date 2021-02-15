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


savepath = path.join(getcwd(), 'data', 'projective_estimations', 'real_machine',
                     '3q_random')
savepath = get_filename(savepath)

nqubits = 3
num_iters = 50
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


# define list of GateObjs which will be our estimate ansatz
# qiskit and qutip use opposite qubit labelling
ansatz = []
ansatz.append(GateObj('U3', 2, True, (params[0], params[1], params[2])))
ansatz.append(GateObj('U3', 1, True, (params[3], params[4], params[5])))
ansatz.append(GateObj('U3', 0, True, (params[6], params[7], params[8])))
ansatz.append(GateObj('CNOT', [2, 1], False))
ansatz.append(GateObj('U3', 2, True, (params[9], params[10], params[11])))
ansatz.append(GateObj('U3', 1, True, (params[12], params[13], params[14])))
ansatz.append(GateObj('CNOT', [2, 1], False))
ansatz.append(GateObj('CNOT', [1, 0], False))
ansatz.append(GateObj('U3', 1, True, (params[15], params[16], params[17])))
ansatz.append(GateObj('U3', 0, True, (params[18], params[19], params[20])))
ansatz.append(GateObj('CNOT', [1, 0], False))
ansatz.append(GateObj('U3', 2, True, (params[21], params[22], params[23])))
ansatz.append(GateObj('U3', 1, True, (params[24], params[25], params[26])))
ansatz.append(GateObj('U3', 0, True, (params[27], params[28], params[29])))


load = True
if load:
    IBMQ.load_account()
provider = IBMQ.get_provider(
    hub='partner-samsung', group='internal', project='imperial')
backend = provider.get_backend('ibmq_singapore')

qreg = QuantumRegister(3, name='qreg')
init_layout = {0: qreg[0], 1: qreg[1], 6: qreg[2]}

q_ansatz = QiskitAnsatz(ansatz, nqubits, backend, init_layout)
qf_ansatz = QiskitFlammiaAnsatz(ansatz, nqubits, backend, init_layout)

zero_est_circ = QiskitEstimator(prob_dist, q_ansatz, num_shots=1024)
proc_est_circ = QiskitFlammiaEstimator(flam_prob_dist, qf_ansatz, num_shots=1024)

zero_ests = []
proc_ests = []
for i in range(num_iters):
    zero_ests.append(zero_est_circ.estimate_zero_fidelity(params, length))
    proc_ests.append(proc_est_circ.estimate_process_fidelity(params, length))

with open(path.join(savepath, "zero_fidelity_estimates.pickle"), 'wb') as f:
    pickle.dump(zero_ests, f)

with open(path.join(savepath, "zero_fidelity_estimates.txt"), 'w') as f:
    for _c in zero_ests:
        f.write(str(_c)+'\n')

with open(path.join(savepath, "process_fidelity_estimates.pickle"), 'wb') as f:
    pickle.dump(zero_ests, f)

with open(path.join(savepath, "process_fidelity_estimates.txt"), 'w') as f:
    for _c in proc_ests:
        f.write(str(_c)+'\n')
