import openfermion as of
from openfermion.transforms import jordan_wigner
import numpy as np
from typing import *
from Fermionic_Gaussian_state import (
    diagonalize_hopping_matrix,
    diagnolize_quadratic_hamiltonian,
    Slater_determinant_decomposition,
    Fermionic_gaussian_states_decomposition
)

from openfermion.linalg import get_sparse_operator
from scipy.linalg import eigh

# Fermionic operator of the hubbard model hamiltonian
# Notice that the chemical potential and magnetic field terms haven't been implemented.

# The FermionOperator class is contained in ops/_fermion_operator.py. In order to support fast addition of 
# FermionOperator instances, the class is implemented as hash table (python dictionary). The keys of the 
# dictionary encode the strings of ladder operators and values of the dictionary store the coefficients.



###################################### Hamiltonian Class ############################################

class Lattice_Hamiltonian:
    def __init__(
        self,
        x_dimension,
        y_dimension,
        periodic=True,
        snake_mapping=True
    ):
        self.x_dim = x_dimension
        self.y_dim = y_dimension
        self.periodic = periodic
        self.snake_mapping = snake_mapping

    def _site_index(self):
        n_site = self.x_dim * self.y_dim

        if self.snake_mapping:
            for i in range(n_site):
                if (i // self.x_dim) % 2 == 0:
                    yield i
                else:
                    q = i // self.x_dim
                    r = i % self.x_dim
                    yield (q + 1) * self.x_dim - r - 1
        else:
            for i in range(n_site):
                yield i

    def _right_index(self, i: int) -> int:
        if self.x_dim == 1:
            return None

        if self.snake_mapping:
        # Even line
            if (i // self.x_dim) % 2 == 0:
                # Boundary site
                if (i + 1) % self.x_dim == 0:
                    if self.periodic:
                        return i + 1 - self.x_dim
                    else:
                        return None
                # Normal site
                else:
                    return i + 1

            # Odd line
            else:
                # Boundary site
                if i % self.x_dim == 0:
                    if self.periodic:
                        return i - 1 + self.x_dim
                    else:
                        return None
                # Normal site
                else:
                    return i - 1
        
        else:
            if (i + 1) % self.x_dim == 0:
                if self.periodic:
                    return i + 1 - self.x_dim
                else:
                    return None
            else:
                return i + 1
    
    def _bottom_index(self, i: int) -> int:
        if self.y_dim == 1:
            return None

        if self.snake_mapping:
            # Even line
            if (i // self.x_dim) % 2 == 0:
                # Boundary site
                if (i // self.x_dim + 1) == self.y_dim:
                    if self.periodic:
                        return i % self.x_dim
                    else:
                        return None
                # Normal site
                else:
                    q = i // self.x_dim
                    r = i % self.x_dim
                    return self.x_dim * (q+2) - r - 1
            
            # Odd line
            else:
                # Boundary site
                if (i // self.x_dim + 1) == self.y_dim:
                    if self.periodic:
                        r = i % self.x_dim
                        return self.x_dim - 1 - r
                    else:
                        return None
                # Normal site
                else:
                    q = i // self.x_dim
                    r = i % self.x_dim
                    return self.x_dim * (q+2) - r - 1
        
        else:
            if i // self.x_dim + 1 == self.y_dim:
                if self.periodic:
                    return i // self.x_dim
                else:
                    return None
            else:
                return i + self.x_dim
        
    def _up_spin_index(self, i):
        return i

    def _down_spin_index(self, i):
        n_site = self.x_dim * self.y_dim
        return i + n_site



class Hubbard_Model(Lattice_Hamiltonian):
    def __init__(
        self,
        x_dimension: int,
        y_dimension: int,
        tunneling: float,
        coulomb: float,
        chemical_potential=0,
        magnetic_field=0,
        periodic=True,
        snake_mapping=True
    ):
        super().__init__(
            x_dimension,
            y_dimension,
            periodic,
            snake_mapping
        )
        self.tunneling = tunneling
        self.coulomb = coulomb
        self.chemical_potential = chemical_potential
        self.magnetic_field = magnetic_field
    
    def fermionic_hamiltonian(self):
    
        n_site = self.x_dim * self.y_dim
        n_spin_orbital = n_site * 2

        vertical_tunneling = of.FermionOperator()
        horizontal_tunneling = of.FermionOperator()
        on_site_interaction = of.FermionOperator()

        for i in self._site_index():
            # right and bottom index
            i_r = self._right_index(i)   
            i_b = self._bottom_index(i)
            
            if self.snake_mapping:
                # Even line
                if (i // self.x_dim) % 2 == 0:
                    # Exclude the boundary case when x_dimension or y_dimension == 2
                    if self.x_dim == 2 and self.periodic and i % 2 == 1:
                        i_r = None

                # Odd line
                else:
                    if self.x_dim == 2 and self.periodic and i % 2 == 0:
                        i_r = None

            else:
                if self.x_dim == 2 and self.periodic and i % 2 == 1:
                    i_r = None
            
            if self.y_dim == 2 and self.periodic and i // self.x_dim + 1 == self.y_dim:
                i_b = None

            # hopping term
            if i_r is not None:
                horizontal_tunneling += self._hopping_term(self._up_spin_index(i), self._up_spin_index(i_r))
                horizontal_tunneling += self._hopping_term(self._down_spin_index(i), self._down_spin_index(i_r))
            
            if i_b is not None:
                vertical_tunneling += self._hopping_term(self._up_spin_index(i), self._up_spin_index(i_b))
                vertical_tunneling += self._hopping_term(self._down_spin_index(i), self._down_spin_index(i_b))

            # on-site interaction
            on_site_interaction += self._on_site_interaction(i)
        
        return horizontal_tunneling, vertical_tunneling, on_site_interaction

    # Here we use the jordan wigner transformation 
    def qubit_hamiltonian(self):
        hamiltonian = sum(self.fermionic_hamiltonian())
        return jordan_wigner(hamiltonian)

    def exact_diagonalization(self):
        matrix_form = get_sparse_operator(self.qubit_hamiltonian()).todense()
        eigenvalues, eigenvectors = eigh(matrix_form)
        return eigenvalues, eigenvectors

    # We need to customize the verital hopping if we want to use the FSWAP network.
    def get_vertical_operator(self):

        vertical_operator_first = of.FermionOperator()
        vertical_operator_second = of.FermionOperator()
        for k in range(self.y_dim-1):
            i = (k+1) * self.x_dim - 1
            if k % 2 == 0:
                vertical_operator_first += self._hopping_term(self._up_spin_index(i), self._up_spin_index(i+1))
                vertical_operator_first += self._hopping_term(self._down_spin_index(i), self._down_spin_index(i+1))
            else:
                vertical_operator_second += self._hopping_term(self._up_spin_index(i), self._up_spin_index(i+1))
                vertical_operator_second += self._hopping_term(self._down_spin_index(i), self._down_spin_index(i+1))

        return vertical_operator_first, vertical_operator_second
    
    def get_ansatz_qubit_operator(self):
        horizontal_tunneling, vertical_tunneling, on_site_interaction = self.fermionic_hamiltonian()
        
        if self.snake_mapping:
            vertical_tunneling = self.get_vertical_operator()
            return jordan_wigner(horizontal_tunneling), \
                   (jordan_wigner(vertical_tunneling[0]), jordan_wigner(vertical_tunneling[1])), \
                    jordan_wigner(on_site_interaction)

        return jordan_wigner(horizontal_tunneling), \
               jordan_wigner(vertical_tunneling), \
               jordan_wigner(on_site_interaction)


    def _hopping_term(self, i, j):
        hopping_term = of.FermionOperator(f'{i}^ {j}', -self.tunneling)
        hopping_term += of.FermionOperator(f'{j}^ {i}', -self.tunneling.conjugate())
        return hopping_term

    def _on_site_interaction(self, i):
        i = self._up_spin_index(i)
        j = self._down_spin_index(i)
        return of.FermionOperator(f'{i}^ {i} {j}^ {j}', self.coulomb)

    

class D_wave_mean_field(Lattice_Hamiltonian):
    
    def __init__(
        self,
        x_dimension: int,
        y_dimension: int,
        tunneling: float,
        superconducting_gap: float,
        periodic=True,
        snake_mapping=True
    ):
        super().__init__(
            x_dimension,
            y_dimension,
            periodic,
            snake_mapping
        )
        self.tunneling = tunneling
        self.delta = superconducting_gap

    def fermionic_hamiltonian(self):

        hamiltonian = of.FermionOperator()

        for i in self._site_index():
            # right and bottom index
            i_r = self._right_index(i)   
            i_b = self._bottom_index(i)
            
            if self.snake_mapping:
                # Even line
                if (i // self.x_dim) % 2 == 0:
                    # Exclude the boundary case when x_dimension or y_dimension == 2
                    if self.x_dim == 2 and self.periodic and i % 2 == 1:
                        i_r = None

                # Odd line
                else:
                    if self.x_dim == 2 and self.periodic and i % 2 == 0:
                        i_r = None

            else:
                if self.x_dim == 2 and self.periodic and i % 2 == 1:
                    i_r = None
            
            if self.y_dim == 2 and self.periodic and i // self.x_dim + 1 == self.y_dim:
                i_b = None

            if i_r is not None:
                # hopping term
                hamiltonian += self._hopping_term(self._up_spin_index(i), self._up_spin_index(i_r))
                hamiltonian += self._hopping_term(self._down_spin_index(i), self._down_spin_index(i_r))
                # mean field term
                hamiltonian += self._mean_field_term(
                    self._up_spin_index(i),
                    self._down_spin_index(i_r),
                    self._down_spin_index(i),
                    self._up_spin_index(i_r),
                    horizontal=True
                )
                        
            if i_b is not None:
                # hopping term
                hamiltonian += self._hopping_term(self._up_spin_index(i), self._up_spin_index(i_b))
                hamiltonian += self._hopping_term(self._down_spin_index(i), self._down_spin_index(i_b))
                # mean field term
                hamiltonian += self._mean_field_term(
                    self._up_spin_index(i),
                    self._down_spin_index(i_b),
                    self._down_spin_index(i),
                    self._up_spin_index(i_b),
                    horizontal=False
                )
        
        return hamiltonian

    def qubit_hamiltonian(self):
        hamiltonian = self.fermionic_hamiltonian()
        return jordan_wigner(hamiltonian)

    def exact_diagonalization(self):
        matrix_form = get_sparse_operator(self.qubit_hamiltonian()).todense()
        eigenvalues, eigenvectors = eigh(matrix_form)
        return eigenvalues, eigenvectors

    def get_matrix_representation(self):
        n_site = 2 * self.x_dim * self.y_dim
        M = np.zeros((n_site, n_site))
        Delta = np.zeros((n_site, n_site))

        for i in self._site_index():
            # right and bottom index
            i_r = self._right_index(i)   
            i_b = self._bottom_index(i)

            if self.snake_mapping:
                # Even line
                if (i // self.x_dim) % 2 == 0:
                    # Exclude the boundary case when x_dimension or y_dimension == 2
                    if self.x_dim == 2 and self.periodic and i % 2 == 1:
                        i_r = None

                # Odd line
                else:
                    if self.x_dim == 2 and self.periodic and i % 2 == 0:
                        i_r = None

            else:
                if self.x_dim == 2 and self.periodic and i % 2 == 1:
                    i_r = None
            
            if self.y_dim == 2 and self.periodic and i // self.x_dim + 1 == self.y_dim:
                i_b = None
            
            if i_r is not None:
                M[self._up_spin_index(i), self._up_spin_index(i_r)] = -self.tunneling
                M[self._up_spin_index(i_r), self._up_spin_index(i)] = -self.tunneling
                M[self._down_spin_index(i), self._down_spin_index(i_r)] = -self.tunneling
                M[self._down_spin_index(i_r), self._down_spin_index(i)] = -self.tunneling

                Delta[self._up_spin_index(i), self._down_spin_index(i_r)] = -self.delta
                Delta[self._up_spin_index(i_r), self._down_spin_index(i)] = -self.delta
                # Delta transpose = -Delta
                Delta[self._down_spin_index(i_r), self._up_spin_index(i)] = self.delta
                Delta[self._down_spin_index(i), self._up_spin_index(i_r)] = self.delta

            if i_b is not None:
                M[self._up_spin_index(i), self._up_spin_index(i_b)] = -self.tunneling
                M[self._up_spin_index(i_b), self._up_spin_index(i)] = -self.tunneling
                M[self._down_spin_index(i), self._down_spin_index(i_b)] = -self.tunneling
                M[self._down_spin_index(i_b), self._down_spin_index(i)] = -self.tunneling

                Delta[self._up_spin_index(i), self._down_spin_index(i_b)] = self.delta
                Delta[self._up_spin_index(i_b), self._down_spin_index(i)] = self.delta
                # Delta transpose = -Delta
                Delta[self._down_spin_index(i_b), self._up_spin_index(i)] = -self.delta
                Delta[self._down_spin_index(i), self._up_spin_index(i_b)] = -self.delta

        return M, Delta

    def get_diagonalization_ops(self):
        M, Delta = self.get_matrix_representation()
        _, W = diagnolize_quadratic_hamiltonian(M, Delta)
        ops_layers, _ = Fermionic_gaussian_states_decomposition(W)
        return ops_layers

    def _hopping_term(self, i, j):
        hopping_term = of.FermionOperator(f'{i}^ {j}', -self.tunneling)
        hopping_term += of.FermionOperator(f'{j}^ {i}', -self.tunneling.conjugate())
        return hopping_term

    def _mean_field_term(self, i_up: int, j_down: int, i_down: int, j_up: int, horizontal: bool):
        delta = self.delta/2 if horizontal else -self.delta/2 
        mean_field_term = of.FermionOperator(f'{i_up}^ {j_down}^', -delta)
        mean_field_term += of.FermionOperator(f'{i_down}^ {j_up}^', delta)
        mean_field_term += of.FermionOperator(f'{j_down} {i_up}', -delta)
        mean_field_term += of.FermionOperator(f'{j_up} {i_down}', delta)
        return mean_field_term

def test_eig():
    hubbard_model = Hubbard_Model(2, 2, 1, 4)
    eig_vals, eig_vecs = hubbard_model.exact_diagonalization()
    print('eigenvalues:', eig_vals)
    print('\n')
    print('eigenvectors:', eig_vecs)


class Noninteracting(Lattice_Hamiltonian):
    pass


if __name__ == '__main__':
    test_eig()