"""
This module contains functions to compute the DWT of a time series
We assume that we compute the partial DWT up to level J
and that the length of the time series is a multiple of 2**J
"""

import matplotlib.pyplot as plt
import numpy as np

from math import ceil, floor

def get_scaling(name):
    """
    Return the coefficients of the scaling filter
    
    Input:
        type name = string
        name = Name of the wavelet filter
    Output:
        type g = 1D numpy array
        g = Vector of coefficients of the scaling filter
    """
    if (name == 'Haar'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/Haar.dat')
    elif (name == 'D4'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/D4.dat')
    elif (name == 'D6'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/D6.dat')
    elif (name == 'D8'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/D8.dat')
    elif (name == 'D10'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/D10.dat')
    elif (name == 'D12'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/D12.dat')
    elif (name == 'D14'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/D14.dat')
    elif (name == 'D16'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/D16.dat')
    elif (name == 'D18'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/D18.dat')
    elif (name == 'D20'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/D20.dat')
    elif (name == 'LA8'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/LA8.dat')
    elif (name == 'LA10'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/LA10.dat')
    elif (name == 'LA12'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/LA12.dat')
    elif (name == 'LA14'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/LA14.dat')
    elif (name == 'LA16'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/LA16.dat')
    elif (name == 'LA18'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/LA18.dat')
    elif (name == 'LA20'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/LA20.dat')
    elif (name == 'C6'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/C6.dat')
    elif (name == 'C12'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/C12.dat')
    elif (name == 'C18'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/C18.dat')
    elif (name == 'C24'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/C24.dat')
    elif (name == 'C30'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/C30.dat')
    elif (name == 'BL14'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/BL14.dat')
    elif (name == 'BL18'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/BL18.dat')
    elif (name == 'BL20'):
        g = np.loadtxt('/Users/ariane/Documents/ResearchProject/wmtsa/scalingcoeff/BL20.dat')
    else:
        raise ValueError('{} has not been implemented yet'.format(name))
    return g

def get_wavelet(g):
    """
    Return the coefficients of the wavelet filter
    
    Input:
        type g = 1D numpy array
        g = Vector of coefficients of the scaling filter
    Output:
        type h = 1D numpy array
        h = Vector of coefficients of the wavelet filter
    """
    L = np.shape(g)[0]
    h = np.zeros(L)
    for l in range(0, L):
        h[l] = ((-1.0) ** l) * g[L - 1 - l]
    return h

def get_WV(h, g, X):
    """
    Level j of pyramid algorithm.
    Take V_(j-1) and return W_j and V_j
    
    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the scaling filter
        type X = 1D numpy array
        X = V_(j-1)
    Output:
        type W = 1D numpy array
        W = W_j
        type V = 1D numpy array
        V = V_j
    """
    N = np.shape(X)[0]
    assert (N % 2 == 0), \
        'Length of vector of scaling coefficients is odd'
    assert (np.shape(h)[0] == np.shape(g)[0]), \
        'Wavelet and scaling filters have different lengths'
    N2 = int(N / 2)
    W = np.zeros(N2)
    V = np.zeros(N2)
    L = np.shape(h)[0]
    for t in range(0, N2):
        for l in range(0, L):
            index = (2 * t + 1 - l) % N
            W[t] = W[t] + h[l] * X[index]
            V[t] = V[t] + g[l] * X[index]
    return (W, V)

def get_X(h, g, W, V):
    """
    Level j of inverse pyramid algorithm.
    Take W_j and V_j and return V_(j-1)
    
    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the scaling filter
        type W = 1D numpy array
        W = W_j
        type V = 1D numpy array
        V = V_j
    Output:
        type X = 1D numpy array
        X = V_(j-1)
    """
    assert (np.shape(W)[0] == np.shape(V)[0]), \
        'Wj and Vj have different lengths'
    assert (np.shape(h)[0] == np.shape(g)[0]), \
        'Wavelet and scaling filters have different lengths'
    N = np.shape(W)[0]
    N2 = int(2 * N)
    X = np.zeros(N2)
    L = np.shape(h)[0]
    for t in range(0, N2):
        for l in range(0, L):
            index1 = (t + l) % N2
            if (index1 % 2 == 1):
                index2 = int((index1 - 1) / 2)
                X[t] = X[t] + h[l] * W[index2] + g[l] * V[index2]
    return X
   
def pyramid(X, name, J):
    """
    Compute the DWT of X up to level J
    
    Input:
        type X = 1D numpy array
        X = Time series which length is a multiple of 2**J
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial DWT
    Output:
        type W = 1D numpy array
        W = Vector of DWT coefficients
    """
    assert (type(J) == int), \
        'Level of DWT must be an integer'
    assert (J >= 1), \
        'Level of DWT must be higher or equal to 1'
    N = np.shape(X)[0]
    assert (N % (2 ** J) == 0), \
        'Length of time series is not a multiple of 2**J'
    g = get_scaling(name)
    h = get_wavelet(g)
    Vj = X
    W = np.zeros(N)
    indb = 0
    for j in range(1, (J + 1)):
        (Wj, Vj) = get_WV(h, g, Vj)
        inde = indb + int(N / (2 ** j))
        W[indb : inde] = Wj
        if (j == J):
            W[inde : N] = Vj
        indb = indb + int(N / (2 ** j))     
    return W

def inv_pyramid(W, name, J):
    """
    Compute the inverse DWT of W up to level J
    
    Input:
        type W = 1D numpy array
        W = Vector of DWT coefficients which length is a multiple of 2**J
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial DWT
    Output:
        type X = 1D numpy array
        X = Original time series
    """
    assert (type(J) == int), \
        'Level of DWT must be an integer'
    assert (J >= 1), \
        'Level of DWT must be higher or equal to 1'
    N = np.shape(W)[0]
    assert (N % (2 ** J) == 0), \
        'Length of vector of DWT coefficients is not a multiple of 2**J'
    g = get_scaling(name)
    h = get_wavelet(g)
    Vj = W[- int(N / (2 ** J)) : ]
    for j in range(J, 0, -1):
        Wj = W[- int(N / (2 ** (j - 1))) : - int(N / 2 ** j)]
        Vj = get_X(h, g, Wj, Vj)
    X = Vj
    return X

def get_DS(X, W, name, J):
    """
    Compute the details and the smooths of the time series
    using the DWT coefficients

    Input:
        type X = 1D numpy array
        X =  Time series which length is a multiple of 2**J
        type W = 1D numpy array
        W = Vector of DWT cofficients which length is a multiple of 2**J
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial DWT
    Output:
        type D = list of 1D numpy arrays (length J)
        D = List of details [D1, D2, ... , DJ]
        type S = list of 1D numpy arrays (length J+1)
        S = List of smooths [S0, S1, S2, ... , SJ]
    """
    assert (type(J) == int), \
        'Level of DWT must be an integer'
    assert (J >= 1), \
        'Level of DWT must be higher or equal to 1'
    assert (np.shape(X)[0] == np.shape(W)[0]), \
        'Time series and vector of DWT coefficients have different length'
    N = np.shape(X)[0]
    assert (N % (2 ** J) == 0), \
        'Length of time series is not a multiple of 2**J'
    # Compute details
    D = []
    for j in range(1, J + 1):
        Wj = np.zeros(N)
        Wj[- int(N / (2 ** (j - 1))) : - int(N / 2 ** j)] = \
            W[- int(N / (2 ** (j - 1))) : - int(N / 2 ** j)]
        Dj = inv_pyramid(Wj, name, J)
        D.append(Dj)
    # Compute smooths
    S = [X]
    for j in range(0, J):
        Sj = S[-1] - D[j]
        S.append(Sj)
    return (D, S)

def NPES(W):
    """
    Compute the normalized partial energy sequence of a time series

    Input:
        type W = 1D numpy array
        W = Time series (or wavelet coefficients)
    Output:
        type C = 1D numpy array
        C = NPES
    """
    N = np.shape(W)[0]
    U = np.flip(np.sort(np.power(np.abs(W), 2.0)), 0)
    C = np.zeros(N)
    for i in range(0, N):
        C[i] = np.sum(U[0 : i + 1]) / np.sum(U)
    return C

def get_nu(name, J):
    """
    Compute the phase shift for LA or coiflet filters

    Input:
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Maximum level for DWT
    Output:
        type nuH = list of J values
        nuH = Shifts for the wavelet filter
        type nuG = list of J values
        nuG = Shifts for the scaling filter
    """
    assert (type(J) == int), \
        'Level of DWT must be an integer'
    assert (J >= 1), \
        'Level of DWT must be higher or equal to 1'
    assert (name[0 : 2] == 'LA' or name[0 : 1] == 'C'), \
        'Wavelet filter must be Daubechies least asymmetric or Coiflet'
    nuH = []
    nuG = []
    # Least asymmetric
    if (name[0 : 2] == 'LA'):
        L = int(name[2 : ])
        if (L == 14):
            nu = int(- L / 2 + 2)
        elif (int(L / 2) % 2 == 0):
            nu = int(- L / 2 + 1)
        else:
            nu = int(- L / 2)
        for j in range(1, J + 1):
            Lj = int((2 ** j - 1) * (L - 1) + 1)
            nuH.append(- int(Lj / 2 + L / 2 + nu - 1))
            nuG.append(int((Lj - 1) * nu / (L - 1)))
    # Coiflet
    else:
        L = int(name[1 :])
        for j in range(1, J + 1):
            Lj = int((2 ** j - 1) * (L - 1) + 1)
            nuH.append(int(- Lj / 2 + L / 6))
            nuG.append(int(- (Lj - 1) * (2 * L - 3) / (3 * (L -1))))
    return (nuH, nuG)

def get_gamma(name, J, N):
    """
    Compute the indices of the last boundary coefficient on the left-side
    and of the last boundary coefficient on the right-side (that is the
    coefficients that are affected by circularity)

    Input:
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Maximum level for DWT
        type N = integer
        N = Length of the time series
    Output:
        type gamHb = list of J values
        gamHb = Indices of the last left-side boundary coefficients (wavelet)
        type gamHe = list of J values
        gamHe = Indices of the first right-side boundary coefficients (wavelet)
        type gamGb = list of J values
        gamGb = Indices of the last left-side boundary coefficients (scaling)
        type gamGe = list of J values
        gamGe = Indices of the first right-side boundary coefficients (scaling)
    """
    (nuH, nuG) = get_nu(name, J)
    if (name[0 : 2] == 'LA'):
        L = int(name[2 : ])
    else:
        L = int(name[1 :])
    gamHb = []
    gamHe = []
    gamGb = []
    gamGe = []
    for j in range(1, J + 1):
        t = int(floor((L - 2) * (1 - 1.0 / (2 ** j))))
        gamHb.append(int((2 ** j * (t + 1) - 1 - abs(nuH[j - 1])) % N))
        gamGb.append(int((2 ** j * (t + 1) - 1 - abs(nuG[j - 1])) % N))
        t = 0
        gamHe.append(int((2 ** j * (t + 1) - 1 - abs(nuH[j - 1])) % N))
        gamGe.append(int((2 ** j * (t + 1) - 1 - abs(nuG[j - 1])) % N))
    return (gamHb, gamHe, gamGb, gamGe)

def get_indices(L, J, N):
    """
    Compute the indices of the values of the details and smooths that
    are affected by circularity

    Input:
        type L = integer
        L = Length of the wavelet filter
        type J = integer
        J = Maximum level for DWT
        type N = integer
        N = Length of the time series
    Output:
        type indb = list of J integers
        indb = Index of last coefficient affected by circularity on the left
        type inde = list of J integers
        inde = Index of last coefficient affected by circularity on the right
    """
    indb = []
    inde = []
    for j in range(1, J + 1):
        Lj = int((2 ** j - 1) * (L - 1) + 1)
        Ljp = int(ceil((L - 2) * (1 - 1 / 2 ** j)))
        indb.append(int(max(2 ** j - 1, (2 ** j) * Ljp - 1, - 1)))
        inde.append(int(min(N + 2 ** j - Lj, N + (2 ** j) * Ljp - Lj, N)))
    return (indb, inde)

def compute_AB(a, j, N):
    """
    Compute the matrix Aj or Bj at level j

    Input:
        type a = 1D numpy array
        a = Scaling filter g for A / wavelet filter h for B
        type j = integer
        j = Level of the pyramid algorithm
        type N = integer
        N = Length of the time series (multiple of 2**j)
    Output:
        type C = Nj * Nj-1 numpy array
        C = Aj or Bj
    """
    assert (type(j) == int), \
        'Level of the pyramid algorithm must be an integer'
    assert (j >= 1), \
        'Level of the pyramid algorithm must be higher or equal to 1'
    assert (N % (2 ** j) == 0), \
        'Length of time series is not a multiple of 2**j'
    L = len(a)
    a0 = np.zeros(N)
    nmax = int(L // N)
    for t in range(0, N):
        for n in range(0, nmax + 1):
            if (t + n * N < L):
                a0[t] = a0[t] + a[t + n * N]
    Nj = int(N / (2 ** j))
    Njm1 = int(N / (2 ** (j - 1)))
    A = np.zeros((Nj, Njm1))
    for t in range(0, Nj):
        for l in range(0, Njm1):
            index = int((2 * t + 1 - l) % Njm1)
            A[t, l] = a0[index]
    return A
