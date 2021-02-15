from typing import List, Union
from qutip import Qobj, sigmax, sigmay, sigmaz, qeye, tensor
from os import path, getcwd, makedirs
import numpy as np
from fidelity_functions import get_pauli_basis


def choose(n, k):
    return np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(np.int(n-k)))


class GateObj:
    """Class representing an abstraction of a quantum gate.

    Designed to be an intermediary between qutip, qiskit and the optimisation algorithm.
    """

    def __init__(self, name: Union[str, List], qubits: Union[int, List],
                 parameterise: bool = None, params: List = None):
        self.name = name
        self.qubits = qubits
        self.parameterise = parameterise
        self.params = params


def U_from_hamiltonian(hamiltonian: List[GateObj], nqubits: int, t: float):
    """Generates the unitary operator from a hamiltonian

    Parameters:
    -----------
    hamiltonian: list of GateObjs defining the Hamiltonian of the system - these should
    always be simple tensor products of Pauli operators with coefficients, CNOTs and
    rotation gates are not supported and should be decomposed before calling this
    nqubits: number of qubits
    t: time for the simulation

    Returns:
    --------
    U_ideal: qutip Qobj of the ideal unitary operator
    """
    exponent = None
    for gate in hamiltonian:
        assert gate.params is not None, "Hamiltonian terms must be supplied with scalar coefficients"
        _op = []
        if isinstance(gate.qubits, int):
            for k in range(nqubits):
                if k == gate.qubits:
                    _op.append(qutip_gate(gate.name))
                else:
                    _op.append(qeye(2))
        else:
            idx = 0
            for k in range(nqubits):
                if k in gate.qubits:
                    _op.append(qutip_gate(gate.name[idx]))
                    idx += 1
                else:
                    _op.append(qeye(2))
        if exponent is None:
            exponent = gate.params*tensor(_op)
        else:
            exponent += gate.params*tensor(_op)
    ideal_U = (-1j*t*exponent).expm()

    return ideal_U


def qutip_gate(gate_name: str):
    """Generates the Pauli gate from a name

    Parameters:
    -----------
    gate_name: string representing the gate, e.g. 'X' for sigmax() etc.
    """
    if gate_name == 'X':
        return sigmax()
    elif gate_name == 'Y':
        return sigmay()
    elif gate_name == 'Z':
        return sigmaz()


def get_filename(filename):
    """ Ensures that a unique filename is used with consequential numbering
    """
    if not path.exists(filename):
        makedirs(filename)
        filename = filename
    else:
        test = False
        idx = 2
        filename += f'_{idx}'
        while not test:
            if not path.exists(filename):
                makedirs(filename)
                filename = filename
                test = True
            else:
                idx += 1
                filename = filename[:-(len(str(idx-1))+1)] + f'_{idx}'
    return filename


def generate_u3(theta, phi, lam):
    u_00 = np.cos(theta/2)
    u_01 = -np.exp(1j*lam)*np.sin(theta/2)
    u_10 = np.exp(1j*phi)*np.sin(theta/2)
    u_11 = np.exp(1j*(lam + phi))*np.cos(theta/2)

    return Qobj([[u_00, u_01], [u_10, u_11]])


def generate_rand_herm(nqubits):
    basis = get_pauli_basis(nqubits)
    ops = np.array(basis)
    coeffs = np.array([2*np.random.rand()-1 for i in range(len(basis))])
    herm_op = np.array([coeffs[i]*ops[i] for i in range(len(coeffs))]).sum(axis=0)

    return herm_op, coeffs
