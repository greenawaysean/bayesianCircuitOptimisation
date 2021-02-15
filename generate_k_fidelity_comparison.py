import numpy as np
from scipy import linalg
import pickle
from os import path, getcwd, makedirs

from fidelity_functions import process_fidelity, k_fidelity, get_pauli_basis, get_state_basis, generate_Bmat
from utils import generate_rand_herm

nqubits = 3
orders = range(nqubits+1)
num_evals = 10000

savefile = path.join(getcwd(), "data", "k_fidelity_comparison", f"{nqubits}_qubits")
if not path.exists(savefile):
    makedirs(savefile)

H, coeffs = generate_rand_herm(nqubits)

# save the coeffs definining the target
with open(path.join(savefile, "target_coeffs.txt"), 'w') as f:
    for c in coeffs:
        f.write(str(c)+'\n')

U = linalg.expm(-1j*H)  # random target unitary

# save the target unitary
with open(path.join(savefile, "target_U.pickle"), 'wb') as f:
    pickle.dump(U, f)

# generate bases and B^-1 so we don't have to later
state_basis = get_state_basis(nqubits)
pauli_basis = get_pauli_basis(nqubits)
Bmats = [generate_Bmat(nqubits, k) for k in orders]

k_fidelities = {k: [] for k in orders}
process_fidelities = {k: [] for k in orders}
for i in range(num_evals):
    _h, _ = generate_rand_herm(nqubits)
    eps = 0.5/num_evals  # rotate to get channels with varying overlaps
    V_rot = linalg.expm(-1j*eps*(i+1)*_h)
    V_rot_dag = np.conj(np.transpose(V_rot))
    V = np.dot(np.dot(V_rot, U), V_rot_dag)
    p_fidel = process_fidelity(U, V, nqubits, basis=pauli_basis)
    for k in orders:
        process_fidelities[k].append(p_fidel)
        k_fidel = k_fidelity(U, V, nqubits, k, Bmat=Bmats[k], basis=state_basis)
        k_fidelities[k].append(k_fidel)

# save data as *.pickle for easy use and *.txt for compatibility
for k in orders:
    with open(path.join(savefile, f"{k}_process_fidelities.pickle"), 'wb') as f:
        pickle.dump(process_fidelities[k], f)

    with open(path.join(savefile, f"{k}_process_fidelities.txt"), 'w') as f:
        for p_fidel in process_fidelities[k]:
            f.write(str(np.real(p_fidel))+'\n')

    with open(path.join(savefile, f"{k}_fidelities.pickle"), 'wb') as f:
        pickle.dump(k_fidelities[k], f)

    with open(path.join(savefile, f"{k}_fidelities.txt"), 'w') as f:
        for k_fidel in k_fidelities[k]:
            f.write(str(np.real(k_fidel))+'\n')
