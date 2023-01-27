import cirq
import numpy as np

def Givens_rot(q0, q1, theta, phi):

    yield cirq.PhasedISwapPowGate(phase_exponent=0.25, exponent=theta*2/np.pi).on(q0, q1)
    yield cirq.rz(phi).on(q1)

def FSWAP(q0, q1):
    circuit = cirq.Circuit(
        cirq.ISWAP(q0, q1),
        cirq.rz(-np.pi/2).on(q0),
        cirq.rz(-np.pi/2).on(q1)
    )
    return circuit

def Pauli_string_rotation(qubits, of_operator, coefficient, theta):

    circuit = cirq.Circuit()

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

    return circuit





# def givens_rotation_matrix(theta, phi):
#     M = np.array([
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0, np.cos(theta/2), -np.exp(1j*phi)*np.sin(theta/2), 0.0],
#         [0.0, np.sin(theta/2),  np.exp(1j*phi)*np.cos(theta/2), 0.0],
#         [0.0, 0.0, 0.0, 1.0]
#     ], dtype=complex)
#     return M

# def control_rx_matrix(theta):
#     M = np.array([
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, np.cos(theta/2), -1j*np.sin(theta/2)],
#         [0.0, 0.0, -1j*np.sin(theta/2), np.cos(theta/2)]
#     ], dtype=complex)
#     return M