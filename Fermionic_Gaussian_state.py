import numpy as np
from numpy.linalg import eig, inv
from copy import deepcopy
from scipy.linalg import schur
import scipy

trunc = 1e-12


def diagonalize_hopping_matrix(M):
    """
    objective:
        To diagonalize the hopping matrix M

    input:
        M (np.array): hopping matrix of Hamiltonain

    return:
        D (np.array): Diagonal matrix of M
        U (np.array): Unitary matrix defines the basis transformation
    """
    eigvals, eigvecs = eig(M)
    D = np.diag(eigvals)
    U = inv(eigvecs)

    return D, U

def schur_antisymmetric_transformation_matrix(N):

    if N % 2 != 0:
        raise ValueError('N must be even!')

    T = np.zeros((N, N))

    for k in range(N):
        if k % 2 == 0:
            T[k//2 ,k] = 1
        elif k % 2 == 1:
            T[(k+N-1)//2, k] = 1
    
    return T

def diagnolize_quadratic_hamiltonian(M, delta):

    # check if the input M and delta satisfy the following properties.
    """
    if M != M.conj().T:
        raise ValueError('A Hermitian matrix M must have the property M = M^dagger')
    
    if delta != -delta.T:
        raise ValueError('Matrix delta must have the property delta = -delta.T')
    """
    
    N = len(M)
    
    I = np.eye(N, dtype=complex)
    Omega = np.zeros((2*N, 2*N), dtype=complex)
    temp_matrix = np.zeros((2*N, 2*N), dtype=complex)

    # Set the matrix Omega
    Omega[:N, :N] = I
    Omega[:N, N:2*N] = I
    Omega[N:2*N, :N] = 1j * I
    Omega[N:2*N, N:2*N] = -1j * I 
    Omega = Omega * 1/np.sqrt(2)

    # Set the matrix of temp_matrix
    temp_matrix[:N, :N] = delta
    temp_matrix[:N, N:2*N] = M
    temp_matrix[N:2*N, :N] = -M.conj()
    temp_matrix[N:2*N, N:2*N] = -delta.conj()

    A = -1j * Omega.conj() @ temp_matrix @ Omega.conj().T
    
    # decompose A throgh a orthogonal transformation R
    
    Sigma, Q = schur(A.real) # firstly, transform A into Schur form

    T = schur_antisymmetric_transformation_matrix(2*N)

    D_ = T @ Sigma @ T.T
    R = T @ Q.T

    W = Omega.conj().T @ R @ Omega
    D = 1j/2 * Omega.T @ D_ @ Omega

    return D, W

def givens_rotation_matrix(x, y, mode=None):

    # G = [[cos(\theta), -e^(i\phi)sin(\theta)],
    #      [sin(\theta),  e^(i\phi)cos(\theta)]] 

    # mode == 'left' -->   a * cos(\theta) + e^(-i\phi) * b * sin(\theta) = 0
    # mode == 'right' -->  a * sin(\theta) + e^(i\phi) * b * cos(\theta) = 0

    if not mode:
        raise ValueError('mode must be either left or right')

    if abs(x) < trunc:

        if mode == 'left':
            cos = 1.0
            sin = 0.0
            phi = 0.0

        elif mode == 'right':
            cos = 0.0
            sin = 1.0
            phi = 0.0

    elif abs(y) < trunc:
            
        if mode == 'left':
            cos = 0.0
            sin = 1.0
            phi = 0.0
        
        elif mode == 'right':
            cos = 1.0
            sin = 0.0
            phi = 0.0

    else:

        phi_x = np.angle(x, deg=False) 
        phi_y = np.angle(y, deg=False)

        if mode == 'left':
            cos = np.absolute(y) / np.sqrt( np.absolute(x)**2 + np.absolute(y)**2 )
            sin = np.absolute(x) / np.sqrt( np.absolute(x)**2 + np.absolute(y)**2 )
            phi = phi_x - phi_y
        
        elif mode == 'right':
            cos = np.absolute(x) / np.sqrt( np.absolute(x)**2 + np.absolute(y)**2 )
            sin = -np.absolute(y) / np.sqrt( np.absolute(x)**2 + np.absolute(y)**2 )
            phi = phi_x - phi_y

    G = np.array([[cos, -np.exp(1j * phi) * sin],
                  [sin,  np.exp(1j * phi) * cos]], dtype=complex)

    return G


def apply_givens_rotation(G, M, i, j, rot_type=None, mode=None):

    if not rot_type:
        raise ValueError('rot_type must be either single or double')

    if not mode:
        raise ValueError('mode must be either left or right')

    if rot_type == 'single':
        
        # G = [[cos(\theta), -e^(i\phi)sin(\theta)],
        #      [sin(\theta),  e^(i\phi)cos(\theta)]] 

        if mode == 'left':
            # If we don't use the deepcopy, the row_i will alter along with M[i, :]
            row_i = deepcopy(M[i, :])        
            row_j = deepcopy(M[j, :])         
            M[i, :] = G[0, 0] * row_i + G[0, 1] * row_j
            M[j, :] = G[1, 0] * row_i + G[1, 1] * row_j

        elif mode == 'right':
            # if the Givens rotation operates on the right side of matrix, 
            # we need to apply G^dagger to the matrix
            G_dagger = G.conj().T
            col_i = deepcopy(M[:, i])
            col_j = deepcopy(M[:, j])
            M[:, i] = col_i * G_dagger[0, 0] + col_j * G_dagger[1, 0]
            M[:, j] = col_i * G_dagger[0, 1] + col_j * G_dagger[1, 1]

    elif rot_type == 'double':
        
        # G = [[cos(\theta), -e^(i\phi)sin(\theta),    0,            0                   ],
        #      [sin(\theta),  e^(i\phi)cos(\theta),    0,            0                   ],
        #      [0,            0,                       cos(\theta), -e(-i\phi)sin(\theta)],
        #      [0,            0,                       sin(\theta),  e(-i\phi)sin(\theta)]] 

        a, b = M.shape

        if mode == 'left':
            
            n = a // 2

            row_i = deepcopy(M[i, :])
            row_j = deepcopy(M[j, :])
            row_i_n = deepcopy(M[i+n, :])
            row_j_n = deepcopy(M[j+n, :])

            M[i, :] = G[0, 0] * row_i + G[0, 1] * row_j
            M[j, :] = G[1, 0] * row_i + G[1, 1] * row_j

            G = G.conj()

            M[i+n, :] = G[0, 0] * row_i_n + G[0, 1] * row_j_n
            M[j+n, :] = G[1, 0] * row_i_n + G[1, 1] * row_j_n

        elif mode == 'right':

            n = b // 2

            G_dagger = G.conj().T
            col_i = deepcopy(M[:, i])
            col_j = deepcopy(M[:, j])
            col_i_n = deepcopy(M[:, i+n])
            col_j_n = deepcopy(M[:, j+n])
            
            M[:, i] = col_i * G_dagger[0, 0] + col_j * G_dagger[1, 0]
            M[:, j] = col_i * G_dagger[0, 1] + col_j * G_dagger[1, 1]

            G_dagger = G_dagger.conj()
            
            M[:, i+n] = col_i_n * G_dagger[0, 0] + col_j_n * G_dagger[1, 0]
            M[:, j+n] = col_i_n * G_dagger[0, 1] + col_j_n * G_dagger[1, 1]

def particle_hole_transformation(W):
    
    m, n = W.shape

    if n != 2 * m:
        raise ValueError('matrix shape n must be 2*m !')
    
    N = m

    col_1 = deepcopy(W[:, N-1])
    col_2 = deepcopy(W[:, 2*N-1])

    W[:, N-1], W[:, 2*N-1] = col_2, col_1

def Slater_determinant_decomposition(U, num_fermion):
    """
    objective:
        To obtain the sequence of givens rotations that decompose the unitary transformation U

    input:
        U (np.array):  Unitary matrix defines the basis transformation
        num_fermion: number of eletrons in our system

    return:
        VQ (np.array): a matrix whose upper right triangle has been zero out by sequence of Givens rotation
    """

    if num_fermion > U.shape[0]:
        raise ValueError('number of fermions must be less than or equal to number of orbitals!')

    Q = np.array(U[:num_fermion, :], dtype=complex)

    m, n = Q.shape

    V = np.eye(m, dtype=complex)

    if m > n:
        raise ValueError('the m*n matrix has wrong input size: m must be less than n')

    # zero out the upper right triangle by applying left givens rotations

    for j in reversed(range(n-m+1, n)):
        for i in range(m - (n-j)):
            # if Q[i, j] is 0 or close to zero, we don't need to zero it out
            if abs(Q[i, j]) > trunc:
                G = givens_rotation_matrix(Q[i, j], Q[i+1, j], mode='left')
                apply_givens_rotation(G, Q, i, i+1, rot_type='single', mode='left')
                apply_givens_rotation(G, V, i, i+1, rot_type='single', mode='left')

    # zero out the remaining terms by applying right givens rotations

    VQU_dagger = deepcopy(Q)    
    
    max_number_of_parallel_rotations = min(m, n-m)

    givens_rotation_layers = []

    # there are total n-1 layers of parallel rotations
    for k in range(n-1):
    
        if m < n-m:   # max_number_of_parallel_rotations = m

            # each time we can parallelize k+1 givens rotations if
            # k+1 <= max_number_of_parallel_rotations
            if k+1 <= max_number_of_parallel_rotations:
                row_indicies = range(k+1)
                col_indicies = range(n-m-k, n-m+k+1, 2)

            elif k+1 > max_number_of_parallel_rotations:

                delta = k+1 - max_number_of_parallel_rotations

                if n-m-k > 0:
                    row_indicies = range(m)
                    col_indicies = range(n-m-k, n-delta, 2)

                else:
                    row_indicies = range(m+k-n+1, m)
                    col_indicies = range(m+k-n+2, n-delta, 2)

        elif m > n-m: # max_number_of_parallel_rotations = n-m

            if k+1 <= max_number_of_parallel_rotations:
                row_indicies = range(k+1)
                col_indicies = range(n-m-k, n-m+k+1, 2)

            elif k+1 > max_number_of_parallel_rotations:
                
                row_start = k+1 - max_number_of_parallel_rotations
                col_start = row_start + 1

                if row_start + max_number_of_parallel_rotations -1 < m:
                    row_indicies = range(row_start, row_start+max_number_of_parallel_rotations)
                    col_indicies = range(col_start, col_start+max_number_of_parallel_rotations*2, 2)

                else:
                    row_indicies = range(row_start, m)
                    col_indicies = range(col_start, col_start+len(row_indicies)*2)

        indicies_to_rotate = zip(row_indicies, col_indicies)
        parallel_rotations_list = []

        for i, j in indicies_to_rotate:

            # M * G^dagger = (G * M^dagger)^dagger
            G = givens_rotation_matrix(VQU_dagger[i,j-1].conj(),
                                       VQU_dagger[i,j].conj(), 
                                       mode='right')

            apply_givens_rotation(G, VQU_dagger, j-1, j, rot_type='single', mode='right')

            theta = np.arccos(G[0, 0].real)   # since dtype(G) = complex
            phi = np.angle(G[1, 1], deg=False)
            parallel_rotations_list.append((j-1, j, theta, phi))

        givens_rotation_layers.append(parallel_rotations_list)

    return givens_rotation_layers, VQU_dagger

def Fermionic_gaussian_states_decomposition(W):
    
    N = len(W) // 2

    # check if W satisfy the constraints
    

    WL = deepcopy(W[N:2*N, :])
    V = np.eye(N, dtype=complex)

    # zero out the upper left triangle of WL
    for j in range(N-1):
        for i in range(N-1-j):
            if abs(WL[i, j]) > trunc:
                G = givens_rotation_matrix(WL[i, j], WL[i+1, j], mode='left')
                apply_givens_rotation(G, WL, i, i+1, rot_type='single', mode='left')
                apply_givens_rotation(G, V, i, i+1, rot_type='single', mode='left')

    # zero out the remaining terms
    VWLU_dagger = deepcopy(WL)

    ops_layers = []

    for k in range(2*N - 1):
        
        parallel_ops_list = []

        # apply particle-hole transformation if k % 2 == 0
        if k % 2 == 0:
            # particle-hole transformation will perform on the Nth column of matrix
            if abs(VWLU_dagger[k // 2, N-1]) > trunc:
                
                parallel_ops_list.append('B')
                particle_hole_transformation(VWLU_dagger)
            
            row_end = k // 2
            col_end = N-2
        
        elif k % 2 == 1:
            row_end = (k - 1) // 2
            col_end = N-1

        if k < N:
            row_start = k
            col_start = N-k-1
        
        elif k >= N:
            row_start = N-1
            col_start = k+1-N

        row_indicies = range(row_start, row_end, -1)
        col_indicies = range(col_start, col_end, 2)

        for i, j in zip(row_indicies, col_indicies):

            if abs(VWLU_dagger[i, j]) > trunc:

                G = givens_rotation_matrix(VWLU_dagger[i, j].conj(), 
                                        VWLU_dagger[i, j+1].conj(),
                                        mode='left')

                apply_givens_rotation(G, VWLU_dagger, j, j+1, rot_type='double', mode='right')

                theta = np.arccos(G[0, 0].real)
                phi = np.angle(G[1, 1], deg=False)
                parallel_ops_list.append((j, j+1, theta, phi))
        
        if parallel_ops_list != []:
            ops_layers.append(parallel_ops_list)

    # use the unitary symmetry to turn the right side of matrix into identity
    for j in reversed(range(N+1, 2*N)):
        for i in range(j-N):
            # if Q[i, j] is 0 or close to zero, we don't need to zero it out
            if abs(VWLU_dagger[i, j]) > trunc:
                G = givens_rotation_matrix(VWLU_dagger[i, j], VWLU_dagger[i+1, j], mode='left')
                apply_givens_rotation(G, VWLU_dagger, i, i+1, rot_type='single', mode='left')
                    
    return ops_layers, VWLU_dagger


def get_neighbor_site_list(m, n, boundary_condiction='periodic'):
    """
    objective:
        To get a list that specifies the index of the neighbor of any vertex in the graph
    """

    neighbor_site_indicies = []

    # the index of the graph is specified as following example
    # 
    #      0   1   2   3   4
    #      5   6   7   8   9
    #      10  11  12  13  14
    #      15  16  17  18  19

    if boundary_condiction == 'periodic':

        for i in range(m*n):

            neighbor_list = []

            neighbor_list.append((i-n) % (m*n))

            neighbor_list.append((i+n) % (m*n))

            if i % n != 0:
                neighbor_list.append(i-1)
            else:
                neighbor_list.append(i-1+n)

            if i % n != n-1:
                neighbor_list.append(i+1)
            else:
                neighbor_list.append(i+1-n)

            neighbor_site_indicies.append(neighbor_list)

    elif boundary_condiction == 'open':

        for i in range(m*n):
            
            neighbor_list = []
            # add the up neighbor
            if i // n != 0:
                neighbor_list.append(i-n)

            # add the down neighbor
            if i // n < m-1:
                neighbor_list.append(i+n)

            # add the left neighbor
            if i % n != 0:
                neighbor_list.append(i-1)

            # add the right neighbor
            if i % n != n-1:
                neighbor_list.append(i+1)

            neighbor_site_indicies.append(neighbor_list)

    return neighbor_site_indicies


def D_wave_mean_field_Hamiltonian(m, n, t=1, mu=0.65, delta=0.3, boundary_condiction='periodic'):
    """
    objective:
        Generate a D wave mean field Hamiltonian of size m x n
    return:
        Numpy matrix M and \Delta that describe the Hamiltonian
    """

    # There are total 2 x m x n number of spin orbitals
    # We encode the index (j, sigma=↑) = j,   (j, sigma=↓) = j+m*n

    neighbor = get_neighbor_site_list(m, n, boundary_condiction=boundary_condiction)
    N = m*n

    t_matrix = np.zeros((N, N))
    delta_matrix_ = np.zeros((N, N))

    for j in range(N):
        
        for k in range(N):
            
            if k in neighbor[j]:
                t_matrix[j, k] = -t
                # horizontal superconducting gap
                if abs(j - k) == 1 or abs(j - k) == n-1:
                    delta_matrix_[j, k] = -delta
                # vertical superconducting gap
                else:
                    delta_matrix_[j, k] = delta
            
            elif j == k:
                t_matrix[j, k] = -mu

    # Matrix indicies are assigned as following
    #      /               |                \
    #      |  (j,↑),(k,↑)  |  (j,↑),(k,↓)   |
    #      |               |                |
    #      |--------------------------------|
    #      |               |                |
    #      |  (j,↓),(k,↑)  |  (j,↓),(k,↓)   |
    #      \               |                /

    M = np.zeros((2*N, 2*N))
    delta_matrix = np.zeros((2*N, 2*N))

    M[:N, :N] = t_matrix
    M[N:2*N, N:2*N] = t_matrix

    delta_matrix[:N, N:2*N] = delta_matrix_
    delta_matrix[N:2*N, :N] = -delta_matrix_

    return M, delta_matrix
    