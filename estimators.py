import copy
import itertools
from typing import List, Union

import numpy as np
from qutip import (Qobj, basis, gate_expand_1toN, qeye,
                   sigmax, sigmay, sigmaz, snot, tensor, rx, ry, rz, cnot)
from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qiskit.circuit import Parameter
from qiskit.compiler import transpile
from utils import GateObj, U_from_hamiltonian, generate_u3
from fidelity_functions import generate_Bmat
from probability_distributions import ProbDist, ChiProbDist, FlammiaProbDist


class QutipAnsatz:
    def __init__(self, nqubits: int, gate_list: List[GateObj] = None, V: Qobj = None):
        self.nqubits = nqubits
        self.gate_list = gate_list
        self.V = V
        self.nqubits = nqubits

    def populate_ansatz(self, params):
        if self.V is not None:
            return self.V
        else:
            idx = 0
            ansatz = tensor([qeye(2)]*self.nqubits)
            for gate in self.gate_list:
                if gate.parameterise:
                    if gate.name == 'U3':
                        u3 = generate_u3(gate.params[idx], gate.params[idx+1],
                                         gate.params[idx+2])
                        gate = gate_expand_1toN(u3, self.nqubits, gate.qubits)
                        ansatz = gate*ansatz
                        idx += 3
                    elif gate.name == 'RX':
                        gate = rx(params[idx], self.nqubits, gate.qubits)
                        ansatz = gate*ansatz
                        idx += 1
                    elif gate.name == 'RY':
                        gate = ry(params[idx], self.nqubits, gate.qubits)
                        ansatz = gate*ansatz
                        idx += 1
                    elif gate.name == 'RZ':
                        gate = rz(params[idx], self.nqubits, gate.qubits)
                        ansatz = gate*ansatz
                        idx += 1
                else:
                    if gate.name == 'X':
                        ansatz = gate_expand_1toN(sigmax(),
                                                  self.nqubits, gate.qubits)*ansatz
                    elif gate.name == 'Y':
                        ansatz = gate_expand_1toN(sigmay(),
                                                  self.nqubits, gate.qubits)*ansatz
                    elif gate.name == 'Z':
                        ansatz = gate_expand_1toN(sigmaz(),
                                                  self.nqubits, gate.qubits)*ansatz
                    elif gate.name == 'CX' or gate.name == 'CNOT':
                        ansatz = cnot(self.nqubits,
                                      gate.qubits[0], gate.qubits[1])*ansatz

            return ansatz


class qCirc:
    """Quantum circuit class

    Separates state preparation, unitary evolution and measurement into separate
    sections and applies them onto a qiskit circuit

    Attributes:
    -----------
    nqubits: number of qubits
    input_state: initial state settings for the simulation
    V: list of GateObj's defining the simulated unitary evolution
    params_list: list of parameters to populate the circuit with

    meas_basis: settings for the measurement basis

    Methods:
    --------
    apply_init_state: input state
    apply_V: (Trotterised) unitary evolution operator, parameterised for optimisation
    rotate_meas_basis: rotate into the eigenbasis of the desired observable
    build_circuit: combine the three independent circuits together
    """

    def __init__(self, nqubits: int, input_state: List[GateObj], V: List[GateObj],
                 meas_basis: List[GateObj], backend: str, init_layout: dict = None):
        self.nqubits = nqubits
        self.input_state = input_state
        self.V = V
        self.meas_basis = meas_basis
        self.backend = backend
        self.init_layout = init_layout
        self.qreg = QuantumRegister(self.nqubits, name='qreg')
        self.creg = ClassicalRegister(self.nqubits)
        self.qc = QuantumCircuit(self.qreg, self.creg)
        self.build_circuit()
        self.qc = transpile(self.qc, backend=self.backend,
                            initial_layout=self.init_layout)

    def populate_circuits(self, params_list):
        circ = copy.deepcopy(self.qc)
        param_dict = self.generate_params_dict(params_list)
        circ = circ.bind_parameters(param_dict)

        return circ

    def generate_params_dict(self, params_list):
        params_dict = {}
        idx = 0
        p_idx = 0
        for _gate in self.V:
            if _gate.parameterise:
                if isinstance(self.params[idx], tuple):
                    for i in range(3):
                        params_dict[self.params[idx][i]] = params_list[p_idx]
                        p_idx += 1
                else:
                    params_dict[self.params[idx]] = params_list[p_idx]
                    p_idx += 1
                idx += 1
        return params_dict

    def apply_init_state(self):
        for _gate in self.input_state:
            apply_gate(self.qc, self.qreg, _gate)

    def apply_V(self):
        params = []
        idx = 0
        for _gate in self.V:
            if _gate.parameterise:
                if _gate.name == 'U3':
                    params.append((Parameter(f'{idx}'),
                                   Parameter(f'{idx + 1}'),
                                   Parameter(f'{idx + 2}')))
                    apply_gate(self.qc, self.qreg, _gate, parameterise=True,
                               param=params[-1])
                    idx += 3
                else:
                    params.append(Parameter(f'{idx}'))
                    apply_gate(self.qc, self.qreg, _gate, parameterise=True,
                               param=params[-1])
                    idx += 1
            else:
                apply_gate(self.qc, self.qreg, _gate)
        self.params = tuple(params)

    def rotate_meas_basis(self):
        meas_idx = []
        for _gate in self.meas_basis:
            apply_gate(self.qc, self.qreg, _gate)
            meas_idx.append(_gate.qubits)
        for i in range(self.nqubits):
            self.qc.measure(self.qreg[i], self.creg[i])

    def build_circuit(self):
        """Builds seperate circuits for input states, observables and unitary
        evolution, with the first two being static and the latter being parameterised.

        The full circuit is built by composing all three circuits together.
        """
        self.apply_init_state()
        self.apply_V()
        self.rotate_meas_basis()


def apply_gate(circ: QuantumCircuit, qreg: QuantumRegister, gate: GateObj,
               parameterise: bool = False, param: Union[Parameter, tuple] = None):
    """Applies a gate to a quantum circuit.

    More complicated gates such as RXX gates should be decomposed into single qubit
    gates and CNOTs prior to calling this function. If parameterise is True, then
    qiskit's placeholder parameter theta will be used in place of any explicit
    parameters.
    """
    if not isinstance(gate.qubits, list):
        q = gate.qubits
        params = gate.params
        if gate.name == 'I':
            pass
        elif gate.name == 'H':
            circ.h(qreg[q])
        elif gate.name == 'HSdag':
            circ.h(qreg[q])
            circ.s(qreg[q])
            circ.h(qreg[q])
        elif gate.name == 'X':
            circ.x(qreg[q])
        elif gate.name == 'Y':
            circ.y(qreg[q])
        elif gate.name == 'Z':
            circ.z(qreg[q])
        elif gate.name == 'RX':
            if parameterise:
                circ.rx(param, qreg[q])
            else:
                circ.rx(params, qreg[q])
        elif gate.name == 'RY':
            if parameterise:
                circ.ry(param, qreg[q])
            else:
                circ.ry(params, qreg[q])
        elif gate.name == 'RZ':
            if parameterise:
                circ.rz(param, qreg[q])
            else:
                circ.rz(params, qreg[q])
        elif gate.name == 'U3':
            if parameterise:
                _params = [i for i in param]
                circ.u3(_params[0], _params[1], _params[2], qreg[q])
            else:
                circ.u3(params[0], params[1], params[2], qreg[q])
    else:
        cntrl = gate.qubits[0]
        trgt = gate.qubits[1]
        circ.cx(qreg[cntrl], qreg[trgt])

    return circ


class QiskitAnsatz:
    def __init__(self, gate_list: List[GateObj], nqubits: int, backend, init_layout):
        self.backend = backend
        self.init_layout = init_layout
        self.gate_list = gate_list
        self.nqubits = nqubits
        self.circs = self.generate_circuits()

    def generate_circuits(self):
        """Builds circuits for all possible combinations of input states and
        observables.

        Returns:
        --------
        circs: dictionary indexing all possible circuits for fidelity estimation
        """
        iters = [''.join(i) for i in itertools.product('0123', repeat=self.nqubits)]
        settings = []
        for x in iters:
            for y in iters:
                settings.append((x, y))
        circs = {}
        for _setting in settings:
            init_state, observ = self.parse_setting(_setting)
            _circ = qCirc(self.nqubits, init_state,
                          self.gate_list, observ, self.backend, self.init_layout)
            circs[_setting] = _circ
        return circs

    def parse_setting(self, setting):
        """Convert setting into a list of GateObj's for easier circuit conversion"""
        _state, _obs = setting
        init_state = []
        for i, _op in enumerate(_state):
            if _op == '0':
                continue
            elif _op == '1':
                _s = GateObj(name='U3', qubits=i, parameterise=True,
                             params=[np.arccos(-1/3), 0.0, 0.0])
            elif _op == '2':
                _s = GateObj(name='U3', qubits=i, parameterise=True,
                             params=[np.arccos(-1/3), 2*np.pi/3, 0.0])
            elif _op == '3':
                _s = GateObj(name='U3', qubits=i, parameterise=True,
                             params=[np.arccos(-1/3), 4*np.pi/3, 0.0])
            init_state.append(_s)
        observe = []
        for i, _op in enumerate(_obs):
            # apply the gates which will rotate the qubits to the req'd basis
            if _op == '0':
                continue
            elif _op == '1':
                _o = GateObj(name='H', qubits=i,
                             parameterise=False, params=None)
            elif _op == '2':
                _o = GateObj(name='HSdag', qubits=i,
                             parameterise=False, params=None)
            elif _op == '3':
                _o = GateObj(name='I', qubits=i,
                             parameterise=False, params=None)
            observe.append(_o)

        return init_state, observe


class QiskitFlammiaAnsatz:
    def __init__(self, gate_list: List[GateObj], nqubits: int, backend, init_layout):
        self.backend = backend
        self.init_layout = init_layout
        self.gate_list = gate_list
        self.nqubits = nqubits
        self.circs = self.generate_circuits()

    def generate_circuits(self):
        iters = [''.join(i) for i in itertools.product('0123', repeat=self.nqubits)]
        settings = []
        for x in iters:
            for y in iters:
                settings.append((x, y))
        bases = [''.join(i) for i in itertools.product('01',
                                                       repeat=len(settings[0][0]))]

        circs = {}
        for _setting in settings:
            for _base in bases:
                _sett = (_setting[0], _setting[1], _base)
                init_state, observ = self.parse_setting(_sett)
                _circ = qCirc(self.nqubits, init_state,
                              self.gate_list, observ, self.backend, self.init_layout)
                circs[_sett] = _circ
        return circs

    def parse_setting(self, setting):
        _state, _obs, _base = setting
        init_state = []
        for i, _op in enumerate(_state):
            _s = None
            if _op == '0':
                if _base[i] == '0':
                    continue
                elif _base[i] == '1':
                    _s = GateObj(name='X', qubits=i,
                                 parameterise=False, params=None)
            elif _op == '1':
                if _base[i] == '0':
                    _s = GateObj(name='U3', qubits=i, parameterise=False,
                                 params=[np.pi/2, 0.0, 0.0])
                elif _base[i] == '1':
                    _s = GateObj(name='U3', qubits=i, parameterise=False,
                                 params=[np.pi/2, np.pi, 0.0])
            elif _op == '2':
                if _base[i] == '0':
                    _s = GateObj(name='U3', qubits=i, parameterise=False,
                                 params=[np.pi/2, np.pi/2, 0.0])
                elif _base[i] == '1':
                    _s = GateObj(name='U3', qubits=i, parameterise=False,
                                 params=[np.pi/2, 3*np.pi/2, 0.0])
            elif _op == '3':
                if _base[i] == '0':
                    continue
                elif _base[i] == '1':
                    _s = GateObj(name='X', qubits=i, parameterise=False)
            if _s is not None:
                init_state.append(_s)

        observe = []
        for i, _op in enumerate(_obs):
            # apply the gates which will rotate the qubits to the req'd basis
            if _op == '0':
                continue
            elif _op == '1':
                _o = GateObj(name='H', qubits=i,
                             parameterise=False, params=None)
            elif _op == '2':
                _o = GateObj(name='HSdag', qubits=i,
                             parameterise=False, params=None)
            elif _op == '3':
                _o = GateObj(name='I', qubits=i,
                             parameterise=False, params=None)
            observe.append(_o)

        return init_state, observe


class QutipEstimator:
    def __init__(self, prob_dist: ProbDist, nqubits: int, ansatz: QutipAnsatz):
        self.prob_dist = prob_dist
        self.prob_dict = self.prob_dist.probabilities
        self.chi_dict = self.prob_dist.chi_dict
        self.probs = [self.prob_dict[key] for key in self.prob_dict]
        self.keys = [key for key in self.prob_dict]
        self.nqubits = nqubits
        self.ansatz = ansatz
        self.input_states, self.meas_bases = self.generate_states_bases()

    def calculate_zf(self, length: int, params: List[float] = None):
        self.length = length
        settings = self.select_settings()
        ideal_chi = [self.chi_dict[i] for i in settings]
        expects = self.evaluate_expectations(settings, params)

        fom = 0
        for i, _chi in enumerate(ideal_chi):
            fom += expects[i] / _chi
        fom += self.length - len(settings)
        fom /= self.length

        return np.real(fom)

    def select_settings(self):
        choices = np.random.choice(
            [i for i in range(len(self.keys))], self.length, p=self.probs, replace=True)
        settings = [self.keys[i] for i in choices]
        settings = [item for item in settings if item[1] != '0' * self.nqubits]

        return settings

    def evaluate_expectations(self, settings: List[str], params: List[float]):
        expectations = []
        for sett in settings:
            init = tensor([Qobj([[1, 0], [0, 0]])]*self.nqubits)
            state = self.input_states[sett[0]] * \
                init*self.input_states[sett[0]].dag()
            basis = self.meas_bases[sett[1]]
            circ = self.ansatz.populate_ansatz(params)

            exp = (circ*state*circ.dag()*basis).tr()
            expectations.append(exp)

        return expectations

    def generate_states_bases(self):
        states = {}
        bases = {}
        for sett in self.keys:
            state = sett[0]
            basis = sett[1]
            states[state] = self.get_input_state(state)
            bases[basis] = self.get_meas_basis(basis)

        return states, bases

    def get_input_state(self, op: str):
        operator = []
        for i in op:
            if i == '0':
                operator.append(qeye(2))
            elif i == '1':
                operator.append(generate_u3(np.arccos(-1/3), 0, 0))
            elif i == '2':
                operator.append(generate_u3(np.arccos(-1/3), 2*np.pi/3, 0))
            elif i == '3':
                operator.append(generate_u3(np.arccos(-1/3), 4*np.pi/3, 0))

        return tensor(operator)

    def get_meas_basis(self, op: str):
        operator = []
        for i in op:
            if i == '0':
                operator.append(qeye(2))
            elif i == '1':
                operator.append(sigmax())
            elif i == '2':
                operator.append(sigmay())
            elif i == '3':
                operator.append(sigmaz())
        return tensor(operator)


class QutipFlammiaEstimator:
    def __init__(self, prob_dist: ProbDist, nqubits: int, ansatz: QutipAnsatz):
        self.prob_dist = prob_dist
        self.prob_dict = self.prob_dist.probabilities
        self.chi_dict = self.prob_dist.chi_dict
        self.probs = [self.prob_dict[key] for key in self.prob_dict]
        self.keys = [key for key in self.prob_dict]
        self.nqubits = nqubits
        self.ansatz = ansatz
        self.input_states, self.meas_bases = self.generate_states_bases()

    def calculate_pf(self, length: int, params: List[float] = None):
        self.length = length
        settings = self.select_settings()
        ideal_chi = [self.chi_dict[(sett[0], sett[1])] for sett in settings]
        expects = self.evaluate_expectations(settings, params)
        evalues = self.generate_evalues(settings)

        fom = 0
        for i, _chi in enumerate(ideal_chi):
            fom += evalues[i]*expects[i] / _chi
        _l = np.int(self.length/2**self.nqubits)
        fom += _l - np.int(len(settings)/2**self.nqubits)
        fom /= _l

        return np.real(fom)

    def select_settings(self):
        choices = np.random.choice(
            [i for i in range(len(self.keys))], np.int(self.length/2**self.nqubits), p=self.probs, replace=True)
        settings = [self.keys[i] for i in choices]
        bases = [''.join(i) for i in itertools.product('01',
                                                       repeat=len(settings[0][0]))]
        new_settings = []
        for sett in settings:
            for base in bases:
                new_settings.append((sett[0], sett[1], base))
        settings = [item for item in new_settings if item[1] != '0' * self.nqubits]

        return settings

    def evaluate_expectations(self, settings, params):
        expectations = []
        for sett in settings:
            state = self.input_states[sett[0], sett[2]]
            basis = self.meas_bases[sett[1]]
            circ = self.ansatz.populate_ansatz(params)

            exp = (circ*state*circ.dag()*basis).tr()
            expectations.append(exp)

        return expectations

    def generate_states_bases(self):
        states = {}
        bases = {}
        all_bases = [''.join(i) for i in itertools.product('01',
                                                           repeat=self.nqubits)]
        for sett in self.keys:
            basis = sett[1]
            bases[basis] = self.get_operator(basis)
            for prj in all_bases:
                states[(sett[0], prj)] = self.get_estate(sett[0], prj)

        return states, bases

    def get_operator(self, op: List[str]):
        operator = []
        for i in op:
            if i == '0':
                operator.append(qeye(2))
            elif i == '1':
                operator.append(sigmax())
            elif i == '2':
                operator.append(sigmay())
            elif i == '3':
                operator.append(sigmaz())

        return tensor(operator)

    def get_estate(self, sigma, estate):
        operator = []
        for i, op in enumerate(sigma):
            if op == '0':
                if estate[i] == '0':
                    _op = basis(2, 0)*basis(2, 0).dag()
                    operator.append(_op)
                elif estate[i] == '1':
                    _op = basis(2, 1)*basis(2, 1).dag()
                    operator.append(_op)
            elif op == '1':
                if estate[i] == '0':
                    _op = 1/np.sqrt(2)*(basis(2, 0) + basis(2, 1))
                    operator.append(_op*_op.dag())
                elif estate[i] == '1':
                    _op = 1/np.sqrt(2)*(basis(2, 0) - basis(2, 1))
                    operator.append(_op*_op.dag())
            elif op == '2':
                if estate[i] == '0':
                    _op = 1/np.sqrt(2)*(basis(2, 0) + 1j*basis(2, 1))
                    operator.append(_op*_op.dag())
                elif estate[i] == '1':
                    _op = 1/np.sqrt(2)*(basis(2, 0) - 1j*basis(2, 1))
                    operator.append(_op*_op.dag())
            elif op == '3':
                if estate[i] == '0':
                    _op = basis(2, 0)*basis(2, 0).dag()
                    operator.append(_op)
                elif estate[i] == '1':
                    _op = basis(2, 1)*basis(2, 1).dag()
                    operator.append(_op)
        return tensor(operator)

    def generate_evalues(self, settings):
        evals = []
        for sett in settings:
            _e = 1
            for i, _op in enumerate(sett[0]):
                if _op == '0':
                    _e *= 1.0
                else:
                    if sett[2][i] == '0':
                        _e *= 1.0
                    else:
                        _e *= -1.0
            evals.append(_e)
        return evals


class QiskitEstimator:
    def __init__(self, prob_dist: ProbDist, ansatz: QiskitAnsatz, num_shots: int,
                 noise_model=None):
        self.prob_dist = prob_dist
        self.prob_dict = self.prob_dist.probabilities
        self.chi_dict = self.prob_dist.chi_dict
        self.probs = [self.prob_dict[key] for key in self.prob_dict]
        self.keys = [key for key in self.prob_dict]
        self.num_shots = num_shots
        self.noise_model = noise_model  # only for Aer simulator
        self.ansatz = ansatz
        self.nqubits = self.ansatz.nqubits
        self.backend = self.ansatz.backend
        self.init_layout = self.ansatz.init_layout
        self.circuits = self.ansatz.circs
        self.quant_inst = QuantumInstance(backend=self.backend, shots=self.num_shots,
                                          initial_layout=self.init_layout,
                                          skip_qobj_validation=False,
                                          noise_model=self.noise_model)

    def estimate_zero_fidelity(self, params, length):
        self.length = length  # how many circuits to include in estimate
        settings, qutip_settings = self.select_settings()
        ideal_chi = [self.chi_dict[i] for i in qutip_settings]
        expects = self.run_circuits(settings, params)

        fidelity = 0
        for i, _chi in enumerate(ideal_chi):
            fidelity += expects[i] / _chi

        fidelity += self.length - len(settings)  # add settings with measurment in 00...0
        fidelity /= self.length

        return np.real(fidelity)  # analytically real, gets rid of some numerical error

    def select_settings(self):
        """Choose a set of settings given a probability distribution"""
        choices = []
        choices = np.random.choice(
            [i for i in range(len(self.keys))], self.length, p=self.probs, replace=True)
        qutip_settings = [self.keys[i] for i in choices]
        # qutip and qiskit use mirrored qubit naming schemes
        settings = []
        for _set in qutip_settings:
            setting0 = _set[0][::-1]
            setting1 = _set[1][::-1]
            settings.append((setting0, setting1))
        # measurements in 00...0 basis always yield +1 expectation value
        settings = [item for item in settings if item[1] != '0' * self.nqubits]
        qutip_settings = [
            item for item in qutip_settings if item[1] != '0' * self.nqubits]
        return settings, qutip_settings

    def run_circuits(self, settings, params):
        """Choose a subset of <length> circuits for fidelity estimation and run them

        Parameters:
        -----------
        params: list of parameters to populate the circuits with (intended to be
        adapted through optimisation)

        Returns:
        --------
        expects: list of expectation values for each circuit in the list
        """
        exec_circs = [self.circuits[qc].populate_circuits(params) for qc in settings]
        results = self.quant_inst.execute(exec_circs, had_transpiled=True)
        q_list = [i for i in range(self.nqubits)][::-1]
        expects = []
        for i, _c in enumerate(settings):
            _ig = []
            for j, _b in enumerate(_c[1]):
                if _b == '0':
                    _ig.append(j)
            _ignore = [q_list[i] for i in _ig]
            expects.append(self.generate_expectation(
                results.get_counts(i), _ignore))

        return expects

    def evaluate_process_zero_fidelities(self, params, num_shots=8192):
        """Evaluate the process and zero fidelities using all measurement settings,
        yielding the highest precision evaluation available under the experimental
        constraints.
        """
        self.quant_inst = QuantumInstance(backend=self.backend, shots=self.num_shots,
                                          initial_layout=self.init_layout,
                                          skip_qobj_validation=False,
                                          noise_model=self.noise_model)
        self.expects = self.run_all_circuits(params)
        perms = [''.join(i) for i in itertools.product('0123', repeat=self.nqubits)]
        self.B_dict = {}
        for i, p in enumerate(perms):
            self.B_dict[p] = i
        process_fidelity = self.evaluate_full_pfidelity()
        zero_fidelity = self.evaluate_full_zfidelity()

        return process_fidelity, zero_fidelity

    def run_all_circuits(self, params):
        chosen_circs = [self.circuits[_s] for _s in self.keys]
        exec_circs = [qc.populate_circuits(params) for qc in chosen_circs]
        results = self.quant_inst.execute(exec_circs, had_transpiled=True)
        q_list = [i for i in range(self.nqubits)][::-1]
        expects = []
        for i, _c in enumerate(self.keys):
            _ig = []
            for j, _b in enumerate(_c[1]):
                if _b == '0':
                    _ig.append(j)
            _ignore = [q_list[i] for i in _ig]
            expects.append(self.generate_expectation(
                results.get_counts(i), _ignore))

        return expects

    def evaluate_full_pfidelity(self):
        d = 2**self.nqubits
        Bmat = generate_Bmat(self.nqubits, self.nqubits)
        F = 0
        chis = [self.chi_dict[key] for key in self.chi_dict]
        chi_keys = [key for key in self.chi_dict]
        keys = [key[0] for key in self.chi_dict]
        for i, _key in enumerate(chi_keys):
            chi = self.chi_dict[_key]
            for j, exp in enumerate(self.expects):
                _set1 = self.B_dict[keys[i]]
                _set2 = self.B_dict[keys[j]]
                F += Bmat[_set1, _set2]*chi*exp

        return F/d**3

    def evaluate_full_zfidelity(self):
        d = 2**self.nqubits
        chis = [self.chi_dict[key] for key in self.chi_dict]
        FOM = 0
        for i, chi in enumerate(chis):
            FOM += chi*self.expects[i]
        return FOM/d**3

    @ staticmethod
    def generate_expectation(counts_dict, ignore=None):
        """Generate the expectation value for a Pauli string operator

        Parameters:
        -----------
        counts_dict: dictionary of counts generated from the machine (or qasm simulator)
        ignore: list of qubits which are not being measured

        Returns:
        --------
        expect: expectation value of the circuit in the measured basis
        """
        if ignore is None:
            ignore = []
        total_counts = 0
        key_len = [len(key) for key in counts_dict]
        N = key_len[0]
        bitstrings = [''.join(i) for i in itertools.product('01', repeat=N)]
        expect = 0
        # add any missing counts to dictionary to avoid errors
        for string in bitstrings:
            if string not in counts_dict:
                counts_dict[string] = 0
            count = 0
            for i, idx in enumerate(string):
                if i in ignore:
                    continue
                if idx == '1':
                    count += 1
            if count % 2 == 0:  # subtract odd product of -ve evalues, add even products
                expect += counts_dict[string]
                total_counts += counts_dict[string]
            else:
                expect -= counts_dict[string]
                total_counts += counts_dict[string]

        return expect / total_counts


class QiskitFlammiaEstimator:
    def __init__(self, prob_dist: ProbDist, ansatz: QiskitAnsatz, num_shots: int,
                 noise_model=None):
        self.prob_dist = prob_dist
        self.prob_dict = self.prob_dist.probabilities
        self.chi_dict = self.prob_dist.chi_dict
        self.probs = [self.prob_dict[key] for key in self.prob_dict]
        self.keys = [key for key in self.prob_dict]
        self.num_shots = num_shots
        self.noise_model = noise_model  # only for Aer simulator
        self.ansatz = ansatz
        self.nqubits = self.ansatz.nqubits
        self.backend = self.ansatz.backend
        self.init_layout = self.ansatz.init_layout
        self.circuits = self.ansatz.circs
        self.quant_inst = QuantumInstance(backend=self.backend, shots=self.num_shots,
                                          initial_layout=self.init_layout,
                                          skip_qobj_validation=False,
                                          noise_model=self.noise_model)

    def estimate_process_fidelity(self, params, length):
        self.length = length
        settings, qutip_settings = self.select_settings()
        expects = self.run_circuits(settings, params)

        ideal_chi = [self.chi_dict[_q] for _q in qutip_settings]

        _expects = []
        idx = 0
        _e = 0
        count = 0
        for i in range(len(settings)):
            _eig = self.generate_eigenvalue(settings[i][0], settings[i][2])
            _e += _eig * expects[i]
            idx += 1
            if idx == 2**self.nqubits:
                count += 1
                _expects.append(_e)
                _e = 0
                idx = 0

        fidelity = 0
        for i, _chi in enumerate(ideal_chi):
            fidelity += _expects[i] / _chi

        fidelity += np.int(self.length/2**self.nqubits) - len(ideal_chi)
        fidelity /= np.int(self.length/2**self.nqubits)

        return np.real(fidelity)

    def parse_setting(self, setting):
        _state, _obs, _base = setting
        init_state = []
        for i, _op in enumerate(_state):
            _s = None
            if _op == '0':
                if _base[i] == '0':
                    continue
                elif _base[i] == '1':
                    _s = GateObj(name='X', qubits=i,
                                 parameterise=False, params=None)
            elif _op == '1':
                if _base[i] == '0':
                    _s = GateObj(name='U3', qubits=i, parameterise=False,
                                 params=[np.pi/2, 0.0, 0.0])
                elif _base[i] == '1':
                    _s = GateObj(name='U3', qubits=i, parameterise=False,
                                 params=[np.pi/2, np.pi, 0.0])
            elif _op == '2':
                if _base[i] == '0':
                    _s = GateObj(name='U3', qubits=i, parameterise=False,
                                 params=[np.pi/2, np.pi/2, 0.0])
                elif _base[i] == '1':
                    _s = GateObj(name='U3', qubits=i, parameterise=False,
                                 params=[np.pi/2, 3*np.pi/2, 0.0])
            elif _op == '3':
                if _base[i] == '0':
                    continue
                elif _base[i] == '1':
                    _s = GateObj(name='X', qubits=i, parameterise=False)
            if _s is not None:
                init_state.append(_s)

        observe = []
        for i, _op in enumerate(_obs):
            # apply the gates which will rotate the qubits to the req'd basis
            if _op == '0':
                continue
            elif _op == '1':
                _o = GateObj(name='H', qubits=i,
                             parameterise=False, params=None)
            elif _op == '2':
                _o = GateObj(name='HSdag', qubits=i,
                             parameterise=False, params=None)
            elif _op == '3':
                _o = GateObj(name='I', qubits=i,
                             parameterise=False, params=None)
            observe.append(_o)

        return init_state, observe

    def select_settings(self):
        # first choose which input states/observables to use
        choices = np.random.choice(
            [i for i in range(len(self.keys))], np.int(self.length/2**self.nqubits), p=self.probs, replace=True)
        qutip_settings = [self.keys[i] for i in choices]

        # next choose which pauli eigenstates to input
        p_choices = np.random.choice(
            [''.join(i) for i in itertools.product('01', repeat=len(self.keys[0][0]))], 2**self.nqubits, replace=False
        )

        # qutip and qiskit use mirrored qubit naming schemes
        settings = []
        for _set in qutip_settings:
            for _pset in p_choices:
                setting0 = _set[0][::-1]
                setting1 = _set[1][::-1]
                setting2 = _pset[::-1]
                settings.append((setting0, setting1, setting2))

        settings = [item for item in settings if item[1] != '0' * self.nqubits]
        qutip_settings = [
            item for item in qutip_settings if item[1] != '0' * self.nqubits]

        return settings, qutip_settings

    def run_circuits(self, settings, params):
        """Choose a subset of <length> circuits for fidelity estimation and run them

        Parameters:
        -----------
        params: list of parameters to populate the circuits with (intended to be
        adapted through optimisation)

        Returns:
        --------
        expects: list of expectation values for each circuit in the list
        """

        chosen_circs = [self.circuits[_s] for _s in settings]
        exec_circs = [qc.populate_circuits(params) for qc in chosen_circs]
        results = self.quant_inst.execute(exec_circs, had_transpiled=True)
        q_list = [i for i in range(self.nqubits)][:: -1]
        expects = []
        for i, _c in enumerate(settings):
            _ig = []
            for j, _b in enumerate(_c[1]):
                if _b == '0':
                    _ig.append(j)
            _ignore = [q_list[i] for i in _ig]
            expects.append(self.generate_expectation(
                results.get_counts(i), _ignore))

        return expects

    def generate_eigenvalue(self, state_in: str, base: str):
        test = True
        for i, _b in enumerate(base):
            if state_in[i] == '0':
                continue
            else:
                if test:
                    if _b == '1':
                        test = False
                else:
                    if _b == '1':
                        test = True
        if test:
            return 1
        else:
            return -1

    @ staticmethod
    def generate_expectation(counts_dict, ignore=None):
        """Generate the expectation value for a Pauli string operator

        Parameters:
        -----------
        counts_dict: dictionary of counts generated from the machine (or qasm simulator)
        ignore: list of qubits which are not being measured

        Returns:
        --------
        expect: expectation value of the circuit in the measured basis
        """
        if ignore is None:
            ignore = []
        total_counts = 0
        key_len = [len(key) for key in counts_dict]
        N = key_len[0]
        bitstrings = [''.join(i) for i in itertools.product('01', repeat=N)]
        expect = 0
        # add any missing counts to dictionary to avoid errors
        for string in bitstrings:
            if string not in counts_dict:
                counts_dict[string] = 0
            count = 0
            for i, idx in enumerate(string):
                if i in ignore:
                    continue
                if idx == '1':
                    count += 1
            if count % 2 == 0:  # subtract odd product of -ve evalues, add even products
                expect += counts_dict[string]
                total_counts += counts_dict[string]
            else:
                expect -= counts_dict[string]
                total_counts += counts_dict[string]

        return expect / total_counts
