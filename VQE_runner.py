import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sympy

import cirq
import tensorflow_quantum as tfq
import tensorflow as tf
from tqdm.notebook import tqdm
from hamiltonians import Hubbard_Model, D_wave_mean_field
from ansatz import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

NUM_EPOCHS = 150

def main():

    hubbard_model = Hubbard_Model(4, 2, 1, 4)
    mean_field_model = D_wave_mean_field(4, 2, 1, 4.180904522613066)

    hamiltonian = hubbard_model.cirq_hamiltonian()

    # ansatz = Hamiltonian_Variational_Ansatz(hubbard_model, repetition=120)
    # ansatz = Symmetry_Breaking_Hamiltonian_Variational_Ansatz(hubbard_model, repetition=30)
    # ansatz = Hardware_Efficient_Ansatz(hubbard_model, repetition=360)
    ansatz = Symmetric_Hardware_Efficient_Ansatz(hubbard_model, repetition=60)
    ansatz_circuit = ansatz.circuit()

    depth = len(cirq.Circuit(ansatz_circuit.all_operations()))
    print('total circuit depth = {}'.format(depth))

    state_preparation_circuit = tfq.convert_to_tensor(
        [State_Preparation(mean_field_model).circuit()]
        # [cirq.Circuit()]
    )


    VQE_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string, name='state_prep_input'),
        tfq.layers.PQC(ansatz_circuit,
                       hamiltonian,
                       differentiator=tfq.differentiators.Adjoint(),
                       initializer=tf.keras.initializers.Constant(value=0))
    ])

    print(f'Initial Energy is {VQE_model(state_preparation_circuit)[0, 0]}')
    opt = tf.keras.optimizers.Adam(0.03)
    energy_history = []
    
    for epoch in range(NUM_EPOCHS):
        with tf.GradientTape() as vqe_tape:
            energy = VQE_model(state_preparation_circuit)

        grads = vqe_tape.gradient(energy, VQE_model.trainable_weights)
        opt.apply_gradients(
            zip(grads, VQE_model.trainable_weights)
        )
        if (epoch + 1) % 5 == 0:
            print(f'Epoch: {epoch + 1}, Total Energy: {energy[0, 0]}')
        energy_history.append(energy[0, 0])

    ED_result = hubbard_model.exact_diagonalization()[0][0]
    print('Exact Ground State Energy', ED_result)

    figure(figsize=(10, 8))
    plt.plot(list(range(NUM_EPOCHS)), energy_history, color='b', marker='x', label="VQE")
    plt.axhline(y=ED_result, color = 'r', linestyle = '-', label="ED_result")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('energy')
    plt.savefig('plot.png')

if __name__ == '__main__':
    main()
    