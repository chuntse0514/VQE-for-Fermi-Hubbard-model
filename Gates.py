import cirq
import numpy as np

def Givens_rot(q0, q1, theta, phi):
    circuit = cirq.Circuit(
        cirq.CX(q1, q0),
        cirq.X(q1),
        cirq.CZ(q0, q1) ** (phi/np.pi),
        cirq.X(q1),
        cirq.rz(-np.pi/2).on(q1),
        cirq.CZ(q0, q1) ** (theta/(np.pi*2)),
        cirq.X(q1),
        cirq.CZ(q0, q1) ** (theta/(np.pi*2)),
        cirq.X(q1),
        cirq.CX(q0, q1) ** (-theta/np.pi),
        cirq.rz(np.pi/2).on(q1),
        cirq.CX(q1, q0)
    )
    return circuit

def FSWAP(q0, q1):
    circuit = cirq.Circuit(
        cirq.ISWAP(q0, q1),
        cirq.rz(-np.pi/2).on(q0),
        cirq.rz(-np.pi/2).on(q1)
    )
    return circuit

def givens_rotation_matrix(theta, phi):
    M = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(theta/2), -np.exp(1j*phi)*np.sin(theta/2), 0.0],
        [0.0, np.sin(theta/2),  np.exp(1j*phi)*np.cos(theta/2), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=complex)
    return M

def control_rx_matrix(theta):
    M = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, np.cos(theta/2), -1j*np.sin(theta/2)],
        [0.0, 0.0, -1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)
    return M