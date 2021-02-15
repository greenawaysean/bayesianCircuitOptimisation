import numpy as np
import itertools
from qutip import sigmax, sigmay, sigmaz, qeye, Qobj, tensor
from scipy import linalg


def get_pauli_basis(nqubits):
    iters = [''.join(i) for i in itertools.product('0123', repeat=nqubits)]
    p_ops = {'0': qeye(2), '1': sigmax(), '2': sigmay(), '3': sigmaz()}
    basis = []
    for item in iters:
        _ops = []
        for k in item:
            _ops.append(p_ops[k])
        basis.append(tensor(_ops).full())
    return basis


def get_state_basis(nqubits):
    iters = [''.join(i) for i in itertools.product('0123', repeat=nqubits)]
    theta = 2*np.arctan(np.sqrt(2))

    A = Qobj([[1.0, 0.0]])
    B = Qobj([[np.cos(theta/2), np.sin(theta/2)]])
    C = Qobj([[np.cos(theta/2), np.exp(1j*2*np.pi/3)*np.sin(theta/2)]])
    D = Qobj([[np.cos(theta/2), np.exp(1j*4*np.pi/3)*np.sin(theta/2)]])

    s_ops = {'0': A.dag()*A,
             '1': B.dag()*B,
             '2': C.dag()*C,
             '3': D.dag()*D
             }

    basis = []
    for item in iters:
        _ops = []
        for k in item:
            _ops.append(s_ops[k])
        basis.append(tensor(_ops).full())
    return basis


def process_fidelity(U, V, nqubits, basis=None):
    if basis is None:
        basis = get_pauli_basis(nqubits)
    d = 2**nqubits
    sum = 0
    for op in basis:
        _rho = np.dot(np.dot(U, op), np.conj(np.transpose(U)))
        _sigma = np.dot(np.dot(V, op), np.conj(np.transpose(V)))
        _trace = np.dot(_rho, _sigma)
        _trace = _trace.diagonal()
        sum += _trace.sum()
    return np.real(sum/d**3)


def zero_fidelity(U, V, nqubits, basis=None):
    if basis is None:
        basis = get_state_basis(nqubits)
    d = 2**nqubits
    sum = 0
    for op in basis:
        _rho = np.dot(np.dot(U, op), np.conj(np.transpose(U)))
        _sigma = np.dot(np.dot(V, op), np.conj(np.transpose(V)))
        _trace = np.dot(_rho, _sigma)
        _trace = _trace.diagonal()
        sum += _trace.sum()
    return np.real(sum/d**2)


def k_fidelity(U, V, nqubits, order, Bmat=None, basis=None):
    if basis is None:
        basis = get_state_basis(nqubits)
    d = 2**nqubits
    if Bmat is None:
        Bmat = generate_Bmat(nqubits, order)
    sum = 0
    for i, op1 in enumerate(basis):
        for j, op2 in enumerate(basis):
            coeff = Bmat[i][j]
            if coeff == 0:
                continue
            _rho = np.dot(np.dot(U, op1), np.conj(np.transpose(U)))
            _sigma = np.dot(np.dot(V, op2), np.conj(np.transpose(V)))
            _trace = np.dot(_rho, _sigma)
            _trace = _trace.diagonal().sum()
            sum += coeff*_trace
    return sum/d**2


def generate_Bmat(nqubits, order):
    A = Qobj([[0.25]*4]*4) - 0.5*qeye(4)
    B_inv = None
    for k in range(order+1):
        s = [qeye(4)]*(nqubits - k) + [A]*k
        Omega_k = None
        track_ops = []
        for i in itertools.permutations(s):
            if i in track_ops:
                continue
            track_ops.append(i)
            if Omega_k is None:
                Omega_k = tensor(list(i))
            else:
                Omega_k += tensor(list(i))
        if B_inv is None:
            B_inv = (-1)**k*Omega_k
        else:
            B_inv += (-1)**k*Omega_k
    return B_inv.full()
