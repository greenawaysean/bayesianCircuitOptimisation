import numpy as np
import sys
from os import path, getcwd, makedirs
import pickle
from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qutip import Qobj, cnot
from utils import GateObj, U_from_hamiltonian, get_filename
from probability_distributions import ChiProbDist
from estimators import QiskitAnsatz, QiskitEstimator, TrueFidelityEst

from GPyOpt_fork.GPyOpt import GPyOpt

sys.path.insert(0, path.join(sys.path[0], 'bayesianCircuitMaster'))

savepath = path.join(getcwd(), 'data', 'BO', 'real_machine',
                     '3q_CNOT_three_applications')
savepath = get_filename(savepath)

ideal_U = cnot(3, 2, 0)
prob_dist = ChiProbDist(nqubits=3, U=ideal_U)
nqubits = 3


# define list of GateObjs which will be our estimate ansatz
ansatz = []

ansatz.append(GateObj('U3', 0, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('U3', 1, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('U3', 2, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('CNOT', [0, 1], False))
ansatz.append(GateObj('CNOT', [1, 2], False))
ansatz.append(GateObj('CNOT', [0, 1], False))
ansatz.append(GateObj('CNOT', [1, 2], False))
ansatz.append(GateObj('U3', 0, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('U3', 1, True, (0.0, 0.0, 0.0)))
ansatz.append(GateObj('U3', 2, True, (0.0, 0.0, 0.0)))

_vals = [0.0]*18

load = True
if load:
    IBMQ.load_account()
provider = IBMQ.get_provider(
    hub='partner-samsung', group='internal', project='imperial')
backend = provider.get_backend('ibmq_singapore')

qreg = QuantumRegister(3, name='qreg')
init_layout = {0: qreg[0], 1: qreg[1], 6: qreg[2]}

q_ansatz = QiskitAnsatz(ansatz, nqubits, backend, init_layout)

est_circ = QiskitEstimator(prob_dist, q_ansatz, num_shots=1024)


def F(params):
    return 1 - est_circ.estimate_zero_fidelity(params[0], length=150)


NB_INIT = 1
NB_ITER = 1
DOMAIN_DEFAULT = [(val - np.pi/4, val + np.pi/4) for i, val in enumerate(_vals)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d}
             for i, d in enumerate(DOMAIN_DEFAULT)]

BO_ARGS_DEFAULT = {'domain': DOMAIN_BO, 'initial_design_numdata': NB_INIT,
                   'model_update_interval': 1, 'hp_update_interval': 5,
                   'acquisition_type': 'LCB', 'acquisition_weight': 5,
                   'acquisition_weight_lindec': True, 'optim_num_anchor': 5,
                   'optimize_restarts': 1, 'optim_num_samples': 10000, 'ARD': False}

myBopt = GPyOpt.methods.BayesianOptimization(f=F, **BO_ARGS_DEFAULT)
myBopt.run_optimization(max_iter=NB_ITER)
myBopt.plot_acquisition(path.join(savepath, 'acquisition_plot.png'))
myBopt.plot_convergence(path.join(savepath, 'convergence_plot.png'))

with open(path.join(savepath, 'model_data.pickle'), 'wb') as f:
    pickle.dump(myBopt, f)

(x_seen, y_seen), (x_exp, y_exp) = myBopt.get_best()

with open(path.join(savepath, 'best_params.pickle'), 'wb') as f:
    pickle.dump(((x_seen, y_seen), (x_exp, y_exp)), f)

nmult = 3  # must be odd

ansatz2 = ansatz*nmult

ideal_U = cnot(3, 0, 2)
prob_dist = ChiProbDist(nqubits=3, U=ideal_U)

q_ansatz = QiskitAnsatz(ansatz2, nqubits, backend, init_layout)

est_circ = QiskitEstimator(prob_dist, q_ansatz, num_shots=512)

unoptimised_pf, unoptimised_zf = est_circ.evaluate_process_zero_fidelities(params=[0.0]*(18*nmult))

print(unoptimised_pf)
print(unoptimised_zf)

best_seen_pf, best_seen_zf = est_circ.evaluate_process_zero_fidelities(params=list(x_seen)*nmult)

print(best_seen_pf)
print(best_seen_zf)

best_exp_pf, best_exp_zf = est_circ.evaluate_process_zero_fidelities(params=list(x_exp)*nmult)

print(best_exp_pf)
print(best_exp_zf)

with open(path.join(savepath, 'fidels.pickle'), 'wb') as f:
    pickle.dump((unoptimised_pf, best_seen_pf, best_exp_pf), f)


with open(path.join(savepath, 'true_FOMs.pickle'), 'wb') as f:
    pickle.dump((unoptimised_zf, best_seen_zf, best_exp_zf), f)
