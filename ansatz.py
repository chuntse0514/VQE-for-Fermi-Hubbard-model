import cirq
import openfermion
from openfermion.transforms import get_quadratic_hamiltonian
from openfermion.circuits import prepare_gaussian_state
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
        self.qubits = hubbard_model.qubits

        self.repetition = repetition
        self.n_site = self.x_dim * self.y_dim
        self.n_qubit = 2 * self.x_dim * self.y_dim

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

class Two_Qubit_Interaction_Ansatz(Ansatz_AbstractClass):
    
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

        params_CZ_rot = sympy.symbols('theta_cz:{}'.format(self.repetition*self.n_qubit*2))
        params_Givens_rot = sympy.symbols('theta_G:{}'.format(self.repetition*self.n_qubit))

        params_theta_1 = sympy.symbols('theta1_rx:{}'.format(self.repetition*self.n_qubit))
        params_theta_2 = sympy.symbols('theta2_ry:{}'.format(self.repetition*self.n_qubit))
        params_theta_3 = sympy.symbols('theta3_rx:{}'.format(self.repetition*self.n_qubit))

        for rep in range(self.repetition):
            theta_CZ_rot = params_CZ_rot[rep*self.n_qubit*2:(rep+1)*self.n_qubit*2]
            theta_Givens_rot = params_Givens_rot[rep*self.n_qubit:(rep+1)*self.n_qubit]
            theta_1 = params_theta_1[rep*self.n_qubit:(rep+1)*self.n_qubit]
            theta_2 = params_theta_2[rep*self.n_qubit:(rep+1)*self.n_qubit]
            theta_3 = params_theta_3[rep*self.n_qubit:(rep+1)*self.n_qubit]
            for i in range(self.n_qubit):
                circuit.append(self.CZ_rot_layer(i, theta_CZ_rot[i:(i+2)]))
            for i in range(self.n_site):
                circuit.append(Givens_rot(self.qubits[i], self.qubits[i+self.n_site], theta_Givens_rot[2*i], theta_Givens_rot[2*i+1]))
            for i in range(self.n_qubit):
                circuit.append(cirq.rx(theta_1[i]).on(self.qubits[i]))
                circuit.append(cirq.ry(theta_2[i]).on(self.qubits[i]))
                circuit.append(cirq.rx(theta_3[i]).on(self.qubits[i]))

        return circuit
    
    def CZ_rot_layer(self, i, params):
        # up spin index
        if i < self.n_site:
            i_r = self._right(i)
            i_b = self._bottom(i)
        # down spin index
        else:
            i_r = self._right(i-self.n_site) 
            i_b = self._bottom(i-self.n_site)
            if i_r is not None:
                i_r = i_r + self.n_site 
            if i_b is not None:
                i_b = i_b + self.n_site 

        yield cirq.CZ(self.qubits[i], self.qubits[i_r]) ** params[0] if i_r is not None else cirq.Circuit()
        yield cirq.CZ(self.qubits[i], self.qubits[i_b]) ** params[1] if i_b is not None else cirq.Circuit()

        
    def _right(self, i):
        return self.hubbard_model._right(i)

    def _bottom(self, i):
        return self.hubbard_model._bottom(i)

    def _up_spin(self, i):
        return self.hubbard_model._up_spin(i)

    def _down_spin(self, i):
        return self.hubbard_model._down_spin(i)

class State_Preparation:
    
    def __init__(
        self,
        mean_field_model: D_wave_mean_field
    ):
        self.mean_field_model = mean_field_model
        self.x_dim = mean_field_model.x_dim
        self.y_dim = mean_field_model.y_dim
        self.snake_mapping = mean_field_model.snake_mapping
        self.qubits = mean_field_model.qubits
        
    def circuit(self):
        quadratic_hamiltonian = get_quadratic_hamiltonian(self.mean_field_model.fermionic_hamiltonian())
        circuit_tree = prepare_gaussian_state(self.qubits, quadratic_hamiltonian)
        circuit = cirq.Circuit(circuit_tree)
        return circuit

if __name__ == '__main__':
    pass