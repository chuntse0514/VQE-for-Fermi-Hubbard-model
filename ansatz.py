import cirq
import openfermion
import numpy as np
import sympy
from hamiltonians import Hubbard_Model, D_wave_mean_field
from Gates import *
import abc

# ToDO: Modify the HVA class such that the snake mapping jw string is compatible with the original HVA ansatz
# 1. ensure whether the FSWAP gate is in the cirq package
# 2. figure out how to implement the swap of vertical hopping
# 3. separate the different circumstance i.e. whether snake mapping or not

class Ansatz_AbstractClass(metaclass=abc.ABCMeta):
    
    def __init__(
        self,
        hubbard_model: Hubbard_Model,
        repetition: int
    ):
        self.x_dim = hubbard_model.x_dim
        self.y_dim = hubbard_model.y_dim
        self.hubbard_model = hubbard_model
        self.snake_mapping = hubbard_model.snake_mapping
        self.hamiltonian = hubbard_model.qubit_hamiltonian()

        self.repetition = repetition
        self.n_site = self.x_dim * self.y_dim
        self.n_qubit = 2 * self.x_dim * self.y_dim
        self.qubits = self.get_qubits()

    def get_qubits(self):

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

    def get_cirq_hamiltonian(self):
        return openfermion.transforms.qubit_operator_to_pauli_sum(self.hamiltonian, self.qubits)

    @abc.abstractmethod
    def circuit(self):
        return NotImplemented


class Hamiltonian_Variational_Ansatz(Ansatz_AbstractClass):

    def __init__(
        self,
        hubbard_model: Hubbard_Model,    # Hubbard_Model object from hamiltonians.py
        repetition: int
    ):  
        super().__init__(
            hubbard_model,
            repetition
        )

    def circuit(self):

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

    def _on_site_layer(self, circuit, jw_on_site, theta):
        for of_operator, coeff in jw_on_site.terms.items():
            circuit.append(Pauli_string_rotation(self.qubits, of_operator, coeff, theta))
                
    def _vertical_layer(self, circuit, jw_vertical, theta):
        if self.snake_mapping:
            jw_vertical_first, jw_vertical_second = jw_vertical
            # implemetation of the even line
            for i in reversed(range(self.x_dim)):
                for j in range(i, self.x_dim):
                    circuit.append(FSWAP(self.qubits[j], self.qubits[j+1]))

                for of_operator, coeff in jw_vertical_first.terms.items():
                    circuit.append(Pauli_string_rotation(self.qubits, of_operator, coeff, theta))
                    
            # implementation of the second line
            for i in range(self.x_dim):
                for j in reversed(range(0, i)):
                    circuit.append(FSWAP(self.qubits[j], self.qubits[j+1]))

                for of_operator, coeff in jw_vertical_second.terms.items():
                    circuit.append(Pauli_string_rotation(self.qubits, of_operator, coeff, theta))
                
        else:
            for of_operator, coeff in jw_vertical.terms.items():
                circuit.append(Pauli_string_rotation(self.qubits, of_operator, coeff, theta))

    def _horizontal_layer(self, circuit, jw_horizontal, theta):
        for of_operator, coeff in jw_horizontal.terms.items():
            circuit.append(Pauli_string_rotation(self.qubits, of_operator, coeff, theta))
    

class Symmetry_Breaking_Hamiltonian_Variational_Ansatz(Ansatz_AbstractClass):

    def __init__(
        self,
        hubbard_model: Hubbard_Model,    # Hubbard_Model object from hamiltonians.py
        repetition: int
    ):  

        super().__init__(
            hubbard_model,
            repetition
        )

    def circuit(self):

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

class Hardware_Efficient_Ansatz(Ansatz_AbstractClass):

    def __init__(
        self,
        hubbard_model: Hubbard_Model,    # Hubbard_Model object from hamiltonians.py
        repetition: int
    ):
        super().__init__(
            hubbard_model,
            repetition
        )

    def circuit(self):

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

class Symmetric_Hardware_Efficient_Ansatz(Ansatz_AbstractClass):

    def __init__(
        self,
        hubbard_model: Hubbard_Model,    # Hubbard_Model object from hamiltonians.py
        repetition: int
    ):
        super().__init__(
            hubbard_model,
            repetition
        )

    def circuit(self):

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
                circuit.append(cirq.rx(theta_rz[i]).on(self.qubits[i]))

            flag = flag * (-1)

        return circuit


class Controlled_Layer_Ansatz(Ansatz_AbstractClass):

    def __init__(
        self,
        hubbard_model: Hubbard_Model,    # Hubbard_Model object from hamiltonians.py
        repetition: int
    ):
        super().__init__(
            hubbard_model,
            repetition
        )

    def circuit(self):

        circuit = cirq.Circuit()
        params_theta_1 = sympy.symbols('theta1_rx:{}'.format(self.repetition*self.n_qubit))
        params_theta_2 = sympy.symbols('theta2_ry:{}'.format(self.repetition*self.n_qubit))
        params_theta_3 = sympy.symbols('theta3_rx:{}'.format(self.repetition*self.n_qubit))

        for rep in range(self.repetition):
            theta_1 = params_theta_1[rep*self.n_qubit:(rep+1)*self.n_qubit]
            theta_2 = params_theta_2[rep*self.n_qubit:(rep+1)*self.n_qubit]
            theta_3 = params_theta_3[rep*self.n_qubit:(rep+1)*self.n_qubit]
            for i in range(self.n_qubit):
                circuit.append(cirq.CZ(self.qubits[i], self.qubits[(i+1) % self.n_qubit]))
            for i in range(self.n_qubit):
                circuit.append(cirq.rx(theta_1[i]).on(self.qubits[i]))
                circuit.append(cirq.ry(theta_2[i]).on(self.qubits[i]))
                circuit.append(cirq.rx(theta_3[i]).on(self.qubits[i]))

        return circuit

class Symmetric_Controlled_Layer_Ansatz(Ansatz_AbstractClass):

    def __init__(
        self,
        hubbard_model: Hubbard_Model,    # Hubbard_Model object from hamiltonians.py
        repetition: int
    ):
        super().__init__(
            hubbard_model,
            repetition
        )

    def circuit(self):

        circuit = cirq.Circuit()
        params_theta_1 = sympy.symbols('theta1_rx:{}'.format(self.repetition*self.n_qubit))
        params_theta_2 = sympy.symbols('theta2_ry:{}'.format(self.repetition*self.n_qubit))
        params_theta_3 = sympy.symbols('theta3_rx:{}'.format(self.repetition*self.n_qubit))
        flag = 1

        for rep in range(self.repetition):
            theta_1 = params_theta_1[rep*self.n_qubit:(rep+1)*self.n_qubit]
            theta_2 = params_theta_2[rep*self.n_qubit:(rep+1)*self.n_qubit]
            theta_3 = params_theta_3[rep*self.n_qubit:(rep+1)*self.n_qubit]
            iterator = range(self.n_qubit) if flag == 1 else reversed(range(self.n_qubit))
            for i in iterator:
                circuit.append(cirq.CZ(self.qubits[i], self.qubits[(i+1) % self.n_qubit]))
            for i in iterator:
                circuit.append(cirq.rx(theta_1[i]).on(self.qubits[i]))
                circuit.append(cirq.ry(theta_2[i]).on(self.qubits[i]))
                circuit.append(cirq.rx(theta_3[i]).on(self.qubits[i]))

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

    def circuit(self):
        ops_layers = self.mean_field_hamiltonian.get_diagonalization_ops()
        circuit = cirq.Circuit()

        for parallel_layer in ops_layers:
            for op in parallel_layer:

                if op == 'B':
                    circuit.append(cirq.X(self.qubits[-1]))

                else:
                    (i0, i1, theta, phi) = op
                    q0, q1 = self.qubits[i0], self.qubits[i1]
                    circuit.append(Givens_rot(q0, q1, theta, phi))

        return circuit