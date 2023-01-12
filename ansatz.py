import cirq
import openfermion
import numpy as np
import sympy
from hamiltonians import Hubbard_Model, D_wave_mean_field

# ToDO: Modify the HVA class such that the snake mapping jw string is compatible with the original HVA ansatz
# 1. ensure whether the FSWAP gate is in the cirq package
# 2. figure out how to implement the swap of vertical hopping
# 3. separate the different circumstance i.e. whether snake mapping or not


class Hamiltonian_Variational_Ansatz:

    def __init__(
        self,
        hubbard_model: Hubbard_Model,    # Hubbard_Model object from hamiltonians.py
        repetition: int
    ):  
        self.x_dim = hubbard_model.x_dim
        self.y_dim = hubbard_model.y_dim
        self.n_site = self.x_dim * self.y_dim
        self.n_qubit = 2 * self.n_site
        self.hubbard_model = hubbard_model
        self.repetition = repetition
        self.snake_mapping = hubbard_model.snake_mapping
        self.qubits = self._get_qubits()
        self.hamiltonian = hubbard_model.qubit_hamiltonian()

    def ansatz_circuit(self):

        # circuit parameters
        circuit = cirq.Circuit()
        params_U = sympy.symbols('theta_U^:{}'.format(self.repetition))
        params_v = sympy.symbols('theta_v^:{}'.format(self.repetition))
        params_h = sympy.symbols('theta_h^:{}'.format(self.repetition))

        # Qubit operator from openfermion
        jw_horizontal, jw_vertical, jw_on_site = self.hubbard_model.get_ansatz_qubit_operator()

        self._on_site_layer(circuit, jw_on_site, params_U[0]/2)
        for i in range(self.repetition):
            self._vertical_layer(circuit, jw_vertical, params_v[i])
            self._horizontal_layer(circuit, jw_horizontal, params_h[i])
            if i != self.repetition - 1:
                self._on_site_layer(circuit, jw_on_site, params_U[i])
            else:
                self._on_site_layer(circuit, jw_on_site, params_U[i] / 2)
        return circuit

    def _get_qubits(self):

        qubits = [0] * self.n_qubit
        for i in range(self.y_dim):
            for j in range(self.x_dim):
                    
                if i % 2 == 1 and self.snake_mapping:
                    k = (i+1) * self.x_dim - j - 1
                else:
                    k = i * self.x_dim + j
                
                qubits[k] = cirq.GridQubit(i, j)
                qubits[k+self.n_site] = cirq.GridQubit(i, j+self.x_dim)

        return qubits

    def _get_cirq_hamiltonian(self):
        return openfermion.transforms.qubit_operator_to_pauli_sum(self.hamiltonian, self.qubits)

    def _on_site_layer(self, circuit, jw_on_site, theta):
        for of_operator, coeff in jw_on_site.terms.items():
            self._pauli_string_rotation(circuit, self.qubits, of_operator, coeff, theta)
                
    def _vertical_layer(self, circuit, jw_vertical, theta):
        if self.snake_mapping:
            jw_vertical_first, jw_vertical_second = jw_vertical
            # implemetation of the even line
            for i in reversed(range(self.x_dim)):
                for j in range(i, self.x_dim):
                    self._FSWAP(circuit, self.qubits, j, j+1)

                for of_operator, coeff in jw_vertical_first.terms.items():
                    self._pauli_string_rotation(circuit, self.qubits, of_operator, coeff, theta)
                    
            # implementation of the second line
            for i in range(self.x_dim):
                for j in reversed(range(0, i)):
                    self._FSWAP(circuit, self.qubits, j, j+1)

                for of_operator, coeff in jw_vertical_second.terms.items():
                    self._pauli_string_rotation(circuit, self.qubits, of_operator, coeff, theta)
                
        else:
            for of_operator, coeff in jw_vertical.terms.items():
                self._pauli_string_rotation(circuit, self.qubits, of_operator, coeff, theta)

    def _horizontal_layer(self, circuit, jw_horizontal, theta):
        for of_operator, coeff in jw_horizontal.terms.items():
            self._pauli_string_rotation(circuit, self.qubits, of_operator, coeff, theta)

    @staticmethod
    def _FSWAP(circuit, qubits, i, j):
        circuit.append(cirq.ISWAP(qubits[i], qubits[j]))
        circuit.append(cirq.rz(-np.pi/2).on(qubits[i]))
        circuit.append(cirq.rz(-np.pi/2).on(qubits[j]))

    @staticmethod
    def _pauli_string_rotation(circuit, qubits, of_operator, coefficient, theta):
        # of_operator is a list like [(1, 'X'), (3, 'Y')]
        if len(of_operator) != 0:
            qubit_indices, pauli_ops = zip(*of_operator)
            
            for i in range(len(pauli_ops)):

                q = qubits[qubit_indices[i]]
                op = pauli_ops[i]

                if i < len(pauli_ops) - 1:
                    if op == 'X':
                        circuit.append(cirq.ry(-np.pi/4).on(q))
                    if op == 'Y':
                        circuit.append(cirq.rx(np.pi/4).on(q))

                    q_next = qubits[qubit_indices[i+1]]
                    circuit.append(cirq.CNOT(q, q_next))
                
                else:
                    if op == 'X':
                        circuit.append(cirq.rx(coefficient * theta).on(q))
                    elif op == 'Y':
                        circuit.append(cirq.ry(coefficient * theta).on(q))
                    else:
                        circuit.append(cirq.rz(coefficient * theta).on(q))
            
            for i in reversed(range(len(pauli_ops))):
                if i > 0:
                    q = qubits[qubit_indices[i]]
                    q_prev = qubits[qubit_indices[i-1]]
                    op = pauli_ops[i-1]
                    circuit.append(cirq.CNOT(q_prev, q))
                    if op == 'X':
                        circuit.append(cirq.ry(np.pi/4).on(q_prev))
                    if op == 'Y':
                        circuit.append(cirq.rx(-np.pi/4).on(q_prev))


class Symmetry_Breaking_Hamiltonian_Variational_Ansatz(Hamiltonian_Variational_Ansatz):

    def __init__(
        self,
        hubbard_model: Hubbard_Model,    # Hubbard_Model object from hamiltonians.py
        repetition: int
    ):  

        super().__init__(
            hubbard_model,
            repetition
        )

    def ansatz_circuit(self):

        # circuit parameters
        circuit = cirq.Circuit()
        params_U = sympy.symbols('theta_U^:{}'.format(self.repetition))
        params_v = sympy.symbols('theta_v^:{}'.format(self.repetition))
        params_h = sympy.symbols('theta_h^:{}'.format(self.repetition))
        params_rx = sympy.symbols('theta_X:{}'.format(self.repetition*self.n_qubit))
        params_ry = sympy.symbols('theta_Y:{}'.format(self.repetition*self.n_qubit))

        # Qubit operator from openfermion
        jw_horizontal, jw_vertical, jw_on_site = self.hubbard_model.get_ansatz_qubit_operator()

        self._on_site_layer(circuit, jw_on_site, params_U[0]/2)
        for i in range(self.repetition):
            self._vertical_layer(circuit, jw_vertical, params_v[i])
            self._horizontal_layer(circuit, jw_horizontal, params_h[i])
            if i != self.repetition - 1:
                self._on_site_layer(circuit, jw_on_site, params_U[i])
            else:
                self._on_site_layer(circuit, jw_on_site, params_U[i] / 2)
            self.HEA_layer(circuit, 
                           params_rx[self.n_qubit*i:self.n_qubit*(i+1)],
                           params_ry[self.n_qubit*i:self.n_qubit*(i+1)])
            
        return circuit


    def HEA_layer(self, circuit, params_rx, params_ry):
        for i in range(self.n_qubit):
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[(i+1) % self.n_qubit]))
            circuit.append(cirq.rx(params_rx[i]).on(self.qubits[i]))
            circuit.append(cirq.ry(params_ry[i]).on(self.qubits[i]))

class Hardware_Efficient_Ansatz(Hamiltonian_Variational_Ansatz):

    def __init__(
        self,
        hubbard_model: Hubbard_Model,    # Hubbard_Model object from hamiltonians.py
        repetition: int
    ):
        super().__init__(
            hubbard_model,
            repetition
        )

    def ansatz_circuit(self):

        circuit = cirq.Circuit()
        params_rx = sympy.symbols('theta_X:{}'.format(self.repetition*self.n_qubit))
        params_ry = sympy.symbols('theta_Y:{}'.format(self.repetition*self.n_qubit))
        params_rz = sympy.symbols('theta_Z:{}'.format(self.repetition*self.n_qubit))

        for rep in range(self.repetition):
            theta_rx = params_rx[self.n_qubit*rep:self.n_qubit*(rep+1)]
            theta_ry = params_ry[self.n_qubit*rep:self.n_qubit*(rep+1)]
            theta_rz = params_rz[self.n_qubit*rep:self.n_qubit*(rep+1)]
            for i in range(self.n_qubit):
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[(i+1) % self.n_qubit]))
                circuit.append(cirq.rx(theta_rx[i]).on(self.qubits[i]))
                circuit.append(cirq.ry(theta_ry[i]).on(self.qubits[i]))
                circuit.append(cirq.rz(theta_rz[i]).on(self.qubits[i]))

        return circuit

class Symmetric_Hardware_Efficient_Ansatz(Hamiltonian_Variational_Ansatz):

    def __init__(
        self,
        hubbard_model: Hubbard_Model,    # Hubbard_Model object from hamiltonians.py
        repetition: int
    ):
        super().__init__(
            hubbard_model,
            repetition
        )

    def ansatz_circuit(self):

        circuit = cirq.Circuit()
        params_rx = sympy.symbols('theta_X:{}'.format(self.repetition*self.n_qubit))
        params_ry = sympy.symbols('theta_Y:{}'.format(self.repetition*self.n_qubit))
        params_rz = sympy.symbols('theta_Z:{}'.format(self.repetition*self.n_qubit))
        flag = 1


        for rep in range(self.repetition):
            theta_rx = params_rx[self.n_qubit*rep:self.n_qubit*(rep+1)]
            theta_ry = params_ry[self.n_qubit*rep:self.n_qubit*(rep+1)]
            theta_rz = params_rz[self.n_qubit*rep:self.n_qubit*(rep+1)]
            iterator = range(self.n_qubit) if flag == 1 else reversed(range(self.n_qubit))
            for i in iterator:
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[(i+1) % self.n_qubit]))
            for i in iterator:
                circuit.append(cirq.rx(theta_rx[i]).on(self.qubits[i]))
                circuit.append(cirq.ry(theta_ry[i]).on(self.qubits[i]))
                circuit.append(cirq.rz(theta_rz[i]).on(self.qubits[i]))

            flag = flag * (-1)

        return circuit


class State_Preparation:
    
    def __init__(
        self,
        mean_field_hamiltonian: D_wave_mean_field
    ):
        self.mean_field_hamiltonian = mean_field_hamiltonian
        self.x_dim = mean_field_hamiltonian.x_dim
        self.y_dim = mean_field_hamiltonian.y_dim
        self.snake_mapping = mean_field_hamiltonian.snake_mapping
        self.n_site = self.x_dim * self.y_dim
        self.n_qubit = 2 * self.n_site
        self.qubits = self._get_qubit()
        

    def _get_qubit(self):
        qubits = [0] * self.n_qubit
        for i in range(self.y_dim):
            for j in range(self.x_dim):
                    
                if i % 2 == 1 and self.snake_mapping:
                    k = (i+1) * self.x_dim - j - 1
                else:
                    k = i * self.x_dim + j
                
                qubits[k] = cirq.GridQubit(i, j)
                qubits[k+self.n_site] = cirq.GridQubit(i, j+self.x_dim)

        return qubits

    def state_preparation_circuit(self):
        ops_layers = self.mean_field_hamiltonian.get_diagonalization_ops()
        circuit = cirq.Circuit()

        for parallel_layer in ops_layers:
            for op in parallel_layer:

                if op == 'B':
                    circuit.append(cirq.X(self.qubits[-1]))

                else:
                    (i1, i2, theta, phi) = op
                    q1, q2 = self.qubits[i1], self.qubits[i2]
                    circuit.append(cirq.CNOT(q2, q1))
                    ### Controlled Ry gate is implemented by Rz(-pi/2) * CRx(·) * Rz(pi/2)
                    ### CRx(·) is implemented by CNOT ** (·/pi)
                    circuit.append(cirq.rz(-np.pi/2).on(q2))
                    circuit.append(cirq.CNOT(q1, q2) ** (2 * theta / np.pi))
                    circuit.append(cirq.rz(np.pi/2).on(q2))
                    circuit.append(cirq.CNOT(q2, q1))
                    circuit.append(cirq.rz(phi).on(q2))

        return circuit

def test_State_Preparation():
    mean_field_hamiltonian = D_wave_mean_field(2, 2, 1, 4)
    state_prep = State_Preparation(mean_field_hamiltonian)
    print(state_prep._get_qubit())
    print(state_prep.state_preparation_circuit())

def test_Hamiltonian_Variational_Ansatz():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow_quantum as tfq
    hubbard_model = Hubbard_Model(2, 2, 1, 4)
    mean_field_hamiltonian = D_wave_mean_field(2, 2, 1, 4)
    ansatz = Hamiltonian_Variational_Ansatz(hubbard_model, repetition=3)
    ansatz_circuit = ansatz.ansatz_circuit()
    hamiltonian = ansatz._get_cirq_hamiltonian()

    tfq.convert_to_tensor([ansatz_circuit])
    print('success!')

def test_tfq():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow_quantum as tfq
    
    q = cirq.GridQubit.rect(1, 5)
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    circuit = cirq.Circuit()

    operator = openfermion.QubitOperator('X0 X4')
    hm = Hubbard_Model(2, 2, 4, 1)
    hva = Hamiltonian_Variational_Ansatz(hm, repetition=2)
    for of_operator, coeff in operator.terms.items():
        print(of_operator)
        print(coeff)
        hva._pauli_string_rotation(circuit, q, of_operator, coeff, a)
    
    print(circuit)

    circuit2 = cirq.Circuit([
        cirq.CNOT(q[0], q[1]),
        cirq.ry(a+b).on(q[1]),
        cirq.CNOT(q[0], q[1])
    ])

    print(circuit2)

    tfq.convert_to_tensor([circuit2])

    print('success!')