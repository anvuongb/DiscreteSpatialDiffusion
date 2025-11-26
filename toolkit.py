import os
import numpy as np
from itertools import product
from scipy.linalg import expm
from scipy.special import logit, expit



def check_for_saved_matrix(r, t_size, size, schedule, PBC=False):
    
    
    if PBC:
        filename = f'data/ftp_{r}_{t_size}_{size}_{schedule}_PBC.npz'
    else:
        filename = f'data/ftp_{r}_{t_size}_{size}_{schedule}.npz'

    if os.path.exists(filename):
        print('Loading pre-computed ftp matrix')
        data = np.load(filename)
        return data['fullTransitionProbability']
    else:
        return None




def get_scheduler(schedule, t_size):
    
    if schedule == 'linear':
        tk = np.linspace(0, 1.0, t_size)
        
    elif schedule == 'quadratic':
        tk = np.linspace(0, 1.0, t_size)**2
        
    elif schedule == 'exp':
       
        t0 = 1e-5
        tk = np.array([0] + [t0 ** (1 - i / (t_size - 1)) for i in range(1, t_size)])

    elif schedule == 'power3':
        tk = np.linspace(0, 1.0, t_size)**3

    elif schedule == 'power4':
        tk = np.linspace(0, 1.0, t_size)**4
        
    elif schedule == 'power5':
        tk = np.linspace(0, 1.0, t_size)**5
    
    elif schedule == 'power6':
        tk = np.linspace(0, 1.0, t_size)**6

    elif schedule == 'power7':
        tk = np.linspace(0, 1.0, t_size)**7
                      
    elif schedule == 'blackout':
        
        k = np.arange(1, t_size)
        
        #tEnd = 7.5
        #tk =  np.hstack((np.array([0,]), - np.log(expit( logit(1-np.exp(-tEnd)) + (k-1)/(t_size-2)*(logit(np.exp(-tEnd))-logit(1-np.exp(-tEnd)) )  ))/tEnd))
        
        tEnd1 = 7.5
        tEnd2 = 2.5
        tk =  np.hstack((np.array([0,]), - np.log(expit( logit(1-np.exp(-tEnd1)) + (k-1)/(t_size-2)*(logit(np.exp(-tEnd1))-logit(1-np.exp(-tEnd2)) )  ))/tEnd2))
        
    elif schedule == 'cosine':
        epsilon = 0.008
        tk = np.linspace(1.0, 0.0, t_size)
        tk = np.cos((tk + epsilon) / (1 + epsilon) * np.pi/2) ** 2
              
      
    else:
        raise NotImplementedError
        

    dtk = np.hstack((np.array([0,]), np.diff(tk)))

    return tk, dtk
    



def get_transition_matrix(r, t_size, size, schedule='quadratic'):
    
    tk, dtk = get_scheduler(schedule=schedule, t_size=t_size)
    
    if ( fullTransitionProbability := check_for_saved_matrix(r, t_size, size, schedule) ) is not None:
        return tk, dtk, fullTransitionProbability
        
    ind2Dto1D = np.arange(size**2).reshape((size, size))
    ind1Dto2D = np.column_stack(np.where(ind2Dto1D >= 0))
    
    transitionRate = np.zeros((size**2, size**2))
    
    for i, j in product(range(size), repeat=2):
        ind0 = ind2Dto1D[i, j]
        if i == 0:
            ind1 = ind2Dto1D[i+1, j]
            transitionRate[ind1, ind0] += r
            transitionRate[ind0, ind0] -= r
        elif i == size-1:
            ind1 = ind2Dto1D[i-1, j]
            transitionRate[ind1, ind0] += r
            transitionRate[ind0, ind0] -= r
        else:
            ind1 = ind2Dto1D[i+1, j]
            transitionRate[ind1, ind0] += r
            ind1 = ind2Dto1D[i-1, j]
            transitionRate[ind1, ind0] += r
            transitionRate[ind0, ind0] -= 2*r
    
        if j == 0:
            ind1 = ind2Dto1D[i, j+1]
            transitionRate[ind1, ind0] += r
            transitionRate[ind0, ind0] -= r
        elif j == size-1:
            ind1 = ind2Dto1D[i, j-1]
            transitionRate[ind1, ind0] += r
            transitionRate[ind0, ind0] -= r
        else:
            ind1 = ind2Dto1D[i, j+1]
            transitionRate[ind1, ind0] += r
            ind1 = ind2Dto1D[i, j-1]
            transitionRate[ind1, ind0] += r
            transitionRate[ind0, ind0] -= 2*r
            

    fullTransitionProbability = np.zeros((size**2, size**2, t_size))
    fullTransitionProbability[..., 0] = np.eye(size**2)
    
                  
    for i in range(1, t_size):
        print(f'solving k={i}')
        buffer = expm(dtk[i]*transitionRate)
        for j in range(size):
            buffer[:, j] /= np.sum(buffer[:, j])  # numerical stability
        fullTransitionProbability[..., i] = fullTransitionProbability[..., i-1].dot(buffer)

    # small numerical instabilities
    fullTransitionProbability[fullTransitionProbability<0] = 0

    np.savez(f'data/ftp_{r}_{t_size}_{size}_{schedule}.npz', 
                fullTransitionProbability=fullTransitionProbability)

    return tk, dtk, fullTransitionProbability


def get_transition_matrix_PBC(r, t_size, size, schedule='quadratic'):
    
    
    tk, dtk = get_scheduler(schedule=schedule, t_size=t_size)
    
    if ( fullTransitionProbability := check_for_saved_matrix(r, t_size, size, schedule, PBC=True) ) is not None:
        return tk, dtk, fullTransitionProbability
        
    ind2Dto1D = np.arange(size**2).reshape((size, size))
    ind1Dto2D = np.column_stack(np.where(ind2Dto1D >= 0))
    
    transitionRate = np.zeros((size**2, size**2))
    
    for i, j in product(range(size), repeat=2):

        ind0 = ind2Dto1D[i, j]

        ind1 = ind2Dto1D[(i+1)%size, j]
        transitionRate[ind1, ind0] += r
        ind1 = ind2Dto1D[(i-1)%size, j]
        transitionRate[ind1, ind0] += r
        ind1 = ind2Dto1D[i, (j+1)%size]
        transitionRate[ind1, ind0] += r
        ind1 = ind2Dto1D[i, (j-1)%size]
        transitionRate[ind1, ind0] += r
        transitionRate[ind0, ind0] -= 4*r
        
    fullTransitionProbability = np.zeros((size**2, 1, t_size))
    bufferState = np.zeros((size**2, 1))
    bufferState[0,0] = 1.
    fullTransitionProbability[0,0,0] = 1.
                      
    for i in range(1, t_size):
        print(f'solving k={i}')
        buffer = expm(dtk[i]*transitionRate)
        for j in range(size**2):
            buffer[:, j] /= np.sum(buffer[:, j])  # numerical stability
        bufferState = buffer.dot(bufferState)
        bufferState /= bufferState.sum()
        fullTransitionProbability[:,0,i] = bufferState[:,0]

    # small numerical instabilities
    fullTransitionProbability[fullTransitionProbability<0] = 0

    np.savez(f'data/ftp_{r}_{t_size}_{size}_{schedule}_PBC.npz', 
                fullTransitionProbability=fullTransitionProbability)

    return tk, dtk, fullTransitionProbability



import numpy as np

from scipy.linalg import expm
# plain version
def laplacian_2d(Nx, Ny):
    N = Nx * Ny
    L = np.zeros((N, N))
    def idx(i, j): return i * Ny + j
    for i in range(Nx):
        for j in range(Ny):
            L[idx(i, j), idx(i, j)] = -4
            L[idx(i, j), idx(i, (j + 1) % Ny)] = 1
            L[idx(i, j), idx(i, (j - 1) % Ny)] = 1
            L[idx(i, j), idx((i + 1) % Nx, j)] = 1
            L[idx(i, j), idx((i - 1) % Nx, j)] = 1
    return L


import numpy as np

# with coding assistance from chatGPT
def laplacian_2d(Nx, Ny):
    N = Nx * Ny
    L = np.zeros((N, N))
    def idx(i, j): return i * Ny + j
    for i in range(Nx):
        for j in range(Ny):
            L[idx(i, j), idx(i, j)] = -4
            L[idx(i, j), idx(i, (j + 1) % Ny)] = 1
            L[idx(i, j), idx(i, (j - 1) % Ny)] = 1
            L[idx(i, j), idx((i + 1) % Nx, j)] = 1
            L[idx(i, j), idx((i - 1) % Nx, j)] = 1
    return L
     

# amazing version
def laplacian_eigenvalues_sparse(Nx, Ny):
    """Create the Laplacian operator directly in Fourier space as a sparse diagonal matrix."""
    kx = np.arange(Nx)
    ky = np.arange(Ny)
    eigenvalues = -4 * (np.sin(np.pi * kx[:, None] / Nx)**2 + np.sin(np.pi * ky[None, :] / Ny)**2)
    return eigenvalues.flatten()  # Create a sparse diagonal matrix

from numpy.fft import fft2, ifft2
def fft2d2vec(v):
    k2 = v.shape[0]
    k = int(k2**0.5)
    vrs = v.reshape(k,k)
    vprs = fft2(vrs,norm="ortho")
    return vprs.reshape(v.shape)

def ifft2d2vec(v):
    k2 = v.shape[0]
    k = int(k2**0.5)
    vrs = v.reshape(k,k)
    vprs = ifft2(vrs,norm="ortho")
    return vprs.reshape(v.shape)

def make_ehat0(k):
    ehat = np.zeros(k**2)
    ehat[0]=1
    return ehat

def translation_vec(k,t):
    ehat = make_ehat0(k)
    Ute = ifft2d2vec(ehat)
    L_fft = laplacian_eigenvalues_sparse(k,k)
    exp_L_fft = np.exp(L_fft*t)
    expLUte = exp_L_fft * Ute # diag matrix is same as element multiply for diag vec
    expLe = UexpLUte = fft2d2vec(expLUte) # get final result
    expLe=np.real_if_close(expLe)
    return expLe

def get_transition_matrix_PBC_fast(r, t_size, size, schedule='quadratic'):
    
    tk, dtk = get_scheduler(schedule=schedule, t_size=t_size)
    
    rtk = r*tk
    
    vectors = []
    for rt in rtk:
        v = translation_vec(k=size,t=rt)
        vectors.append(v)
        
    stacked = np.stack(vectors,axis=-1)

    fullTransitionProbability = stacked[:,None,:]
    # small numerical instabilities
    fullTransitionProbability[fullTransitionProbability<0] = 0

    return tk, dtk, fullTransitionProbability
