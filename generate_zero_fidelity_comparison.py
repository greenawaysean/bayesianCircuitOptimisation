import numpy as np
from scipy import linalg
import pickle
from os import path, getcwd, makedirs

from fidelity_functions import process_fidelity, k_fidelity, zero_fidelity, get_pauli_basis, get_state_basis, generate_Bmat
from utils import generate_rand_herm

qubits = [2, 3, 5]
num_evals = 10000

savefile = path.join(getcwd(), "data", "zero_fidelity_comparison")
if not path.exists(savefile):
    makedirs(savefile)

for nqubits in qubits:
    H, coeffs = generate_rand_herm(nqubits)
    # save the coeffs definining the target
    with open(path.join(savefile, f"{nqubits}_qubits_target_coeffs.txt"), 'w') as f:
        for c in coeffs:
            f.write(str(c)+'\n')

    U = linalg.expm(-1j*H)  # random target unitary

    # save the target unitary
    with open(path.join(savefile, f"{nqubits}_qubits_target_U.pickle"), 'wb') as f:
        pickle.dump(U, f)

    # generate bases and B^-1 so we don't have to later
    state_basis = get_state_basis(nqubits)
    pauli_basis = get_pauli_basis(nqubits)

    zero_fidelities = []
    process_fidelities = []
    for i in range(num_evals):
        _h, _ = generate_rand_herm(nqubits)
        eps = 1/num_evals  # rotate to get channels with varying overlaps
        V_rot = linalg.expm(-1j*eps*i*_h)
        V_rot_dag = np.conj(np.transpose(V_rot))
        V = np.dot(np.dot(V_rot, U), V_rot_dag)
        p_fidel = process_fidelity(U, V, nqubits, basis=pauli_basis)
        process_fidelities.append(p_fidel)
        zero_fidel = zero_fidelity(U, V, nqubits, basis=state_basis)
        zero_fidelities.append(zero_fidel)

    # save data as *.pickle for easy use and *.txt for compatibility
    with open(path.join(savefile, f"{nqubits}_q_process_fidelities.pickle"), 'wb') as f:
        pickle.dump(process_fidelities, f)

    with open(path.join(savefile, f"{nqubits}_q_process_fidelities.txt"), 'w') as f:
        for p_fidel in process_fidelities:
            f.write(str(np.real(p_fidel))+'\n')

    with open(path.join(savefile, f"{nqubits}_q_zero_fidelities.pickle"), 'wb') as f:
        pickle.dump(zero_fidelities, f)

    with open(path.join(savefile, f"{nqubits}_q_zero_fidelities.txt"), 'w') as f:
        for zero_fidel in zero_fidelities:
            f.write(str(np.real(zero_fidel))+'\n')
