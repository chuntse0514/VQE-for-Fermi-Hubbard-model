import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import tensorflow as tf
import tensorflow_quantum as tfq
from hamiltonians import *
from ansatz import *

def optimize_mean_field_parameter(x_dim, y_dim):
    hubbard_model = Hubbard_Model(3, 2, 1, 4, snake_mapping=True)
    ansatz = Symmetric_Hardware_Efficient_Ansatz(hubbard_model, repetition=2)
    ansatz_circuit = ansatz.circuit()
    hamiltonian = hubbard_model.cirq_hamiltonian()

    delta_list = np.linspace(0, 8, 200)
    energy_list = []

    for delta in delta_list:
      
        mean_field_model = D_wave_mean_field(x_dim, y_dim, 1, delta)
        state_preparation_circuit = tfq.convert_to_tensor(
          [State_Preparation(mean_field_model).circuit()]
        )
        VQE_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string, name='state_prep_input'),
        tfq.layers.PQC(ansatz_circuit,
                        hamiltonian,
                        differentiator=tfq.differentiators.Adjoint(),
                        initializer=tf.keras.initializers.Constant(value=0))
        ])
        energy = VQE_model(state_preparation_circuit)[0, 0]
        energy_list.append(energy)

    min_value = min(energy_list)
    min_index = energy_list.index(min_value)
    print('Minimum Mean Field Energy {}, Corresponding Delta = {}'.format(min_value, delta_list[min_index]))

    figure(figsize=(20, 8))
    plt.plot(delta_list, energy_list, marker='x', color='b')
    plt.show()

optimize_mean_field_parameter(3, 2)