import numpy as np
from qutip import (sigmax, sigmay, sigmaz, qeye, basis, gate_expand_1toN,
                   Qobj, tensor, snot)
from scipy import linalg
from scipy.sparse import csc_matrix
import copy
import itertools
from utils import generate_u3


class ProbDist:
    """Base class for generating probability distributions for fidelity estimation
    """

    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.pauli_strings = self.pauli_permutations()

    def get_probabilities(self):
        raise NotImplementedError

    def pauli_permutations(self):
        return [''.join(i) for i in
                itertools.product('0123', repeat=self.nqubits)]


class ChiProbDist(ProbDist):
    """Probability distribution for estimating the 0-fidelity based on an
    adaptation of 10.1103/PhysRevLett.106.230501
    """

    def __init__(self, nqubits: int, U: Qobj):
        super().__init__(nqubits)
        self.tens_ops = self.get_tensored_ops()
        self.tens_states = self.get_tensored_states()
        self.U = U.full()
        self.probabilities, self.chi_dict = self.get_probs_and_chis()

    def get_probs_and_chis(self):
        d = 2**self.nqubits
        input_states, observables = self.generate_states_observables()
        probabilities = {}
        chi_dict = {}
        for _state_idx in self.pauli_strings:
            for _obs_idx in self.pauli_strings:
                _state = input_states[_state_idx]
                _obs = observables[_obs_idx]
                _trace = np.dot(self.U, _state)
                _trace = np.dot(_trace, np.conj(np.transpose(self.U)))
                _trace = np.dot(_trace, _obs)
                _trace = _trace.diagonal()
                chi = _trace.sum(axis=0)
                chi_dict[_state_idx, _obs_idx] = chi
                probabilities[(_state_idx, _obs_idx)] = (
                    1/d**3)*np.real(chi)**2
        return probabilities, chi_dict

    def generate_states_observables(self):
        init_state = tensor([basis(2, 0)] * self.nqubits).full()
        input_states = {}
        observables = {}
        for i, _op in enumerate(self.tens_ops):
            _state = self.pauli_strings[i]
            _input_state = copy.deepcopy(self.tens_states[i])
            observables[_state] = copy.deepcopy(_op)
            _init_copy = copy.deepcopy(init_state)
            state = np.dot(_input_state, _init_copy)
            input_states[_state] = np.dot(state, np.conj(np.transpose(state)))

        return input_states, observables

    def get_tensored_ops(self):
        tens_ops = []
        for _state in self.pauli_strings:
            _ops = []
            for i in _state:
                if i == '0':
                    _ops.append(qeye(2))
                if i == '1':
                    _ops.append(sigmax())
                if i == '2':
                    _ops.append(sigmay())
                if i == '3':
                    _ops.append(sigmaz())
            _op = tensor(_ops)
            tens_ops.append(_op.full())

        return tens_ops

    def get_tensored_states(self):
        tens_ops = []
        for _state in self.pauli_strings:
            _ops = []
            for i in _state:
                if i == '0':
                    _ops.append(qeye(2))
                if i == '1':
                    _ops.append(generate_u3(np.arccos(-1/3), 0, 0))
                if i == '2':
                    _ops.append(generate_u3(np.arccos(-1/3), 2*np.pi/3, 0))
                if i == '3':
                    _ops.append(generate_u3(np.arccos(-1/3), 4*np.pi/3, 0))
            _op = tensor(_ops)
            tens_ops.append(_op.full())

        return tens_ops


class FlammiaProbDist(ProbDist):
    """Probability distribution for estimating the process fidelity as in
       10.1103/PhysRevLett.106.230501
    """

    def __init__(self, nqubits: int, U: Qobj):
        super().__init__(nqubits)
        self.tens_ops = self.get_tensored_ops()
        self.tens_states = self.get_tensored_states()
        self.U = U.full()
        self.probabilities, self.chi_dict = self.get_probs_and_chis()

    def get_probs_and_chis(self):
        d = 2**self.nqubits
        input_states, observables = self.generate_states_observables()
        probabilities = {}
        chi_dict = {}
        for _state_idx in self.pauli_strings:
            for _obs_idx in self.pauli_strings:
                _state = input_states[_state_idx]
                _obs = observables[_obs_idx]
                _trace = np.dot(self.U, _state)
                _trace = np.dot(_trace, np.conj(np.transpose(self.U)))
                _trace = np.dot(_trace, _obs)
                _trace = _trace.diagonal()
                chi = _trace.sum(axis=0)
                chi_dict[_state_idx, _obs_idx] = chi  # np.real(chi)
                probabilities[(_state_idx, _obs_idx)] = np.abs((1/d**4)*chi**2)
        return probabilities, chi_dict

    def generate_states_observables(self):
        input_states = {}
        observables = {}
        for i, _op in enumerate(self.tens_ops):
            _state = self.pauli_strings[i]
            _input_state = copy.deepcopy(_op)
            observables[_state] = copy.deepcopy(_op)
            input_states[_state] = _input_state

        return input_states, observables

    def get_tensored_ops(self):
        tens_ops = []
        for _state in self.pauli_strings:
            _ops = []
            for i in _state:
                if i == '0':
                    _ops.append(qeye(2))
                if i == '1':
                    _ops.append(sigmax())
                if i == '2':
                    _ops.append(sigmay())
                if i == '3':
                    _ops.append(sigmaz())
            _op = tensor(_ops)
            tens_ops.append(_op.full())

        return tens_ops

    def get_tensored_states(self):
        tens_ops = []
        for _state in self.pauli_strings:
            _ops = []
            for i in _state:
                if i == '0':
                    _ops.append(qeye(2))
                if i == '1':
                    _ops.append(sigmax())
                if i == '2':
                    _ops.append(sigmay())
                if i == '3':
                    _ops.append(sigmaz())
            _op = tensor(_ops)
            tens_ops.append(_op.full())

        return tens_ops
