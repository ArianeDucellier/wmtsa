"""
This module contains functions to compute the MODWT of a time series
"""

import matplotlib.pyplot as plt
import numpy as np

from math import sqrt

import DWT

def get_scaling(name):
    """
    Return the coefficients of the MODWT scaling filter
    
    Input:
        type name = string
        name = Name of the wavelet filter
    Output:
        type g = 1D numpy array
        g = Vector of coefficients of the MODWT scaling filter
    """
    g = DWT.get_scaling(name)
    g = g / sqrt(2.0)
    return g

def get_wavelet(g):
    """
    Return the coefficients of the MODWT wavelet filter
    
    Input:
        type g = 1D numpy array
        g = Vector of coefficients of the MODWT scaling filter
    Output:
        type h = 1D numpy array
        h = Vector of coefficients of the MODWT wavelet filter
    """
    h = DWT.get_wavelet(g)
    return h

def get_WV(h, g, j, X):
    """
    Level j of pyramid algorithm.
    Take V_(j-1) and return W_j and V_j
    
    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the MODWT wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the MODWT scaling filter
        type j = integer
        j = Current level of the pyramid algorithm
        type X = 1D numpy array
        X = V_(j-1)
    Output:
        type W = 1D numpy array
        W = W_j
        type V = 1D numpy array
        V = V_j
    """
    assert (np.shape(h)[0] == np.shape(g)[0]), \
        'Wavelet and scaling filters have different lengths'
    N = np.shape(X)[0]
    W = np.zeros(N)
    V = np.zeros(N)
    L = np.shape(h)[0]
    for t in range(0, N):
        for l in range(0, L):
            index = (t - (2 ** (j - 1)) * l) % N
            W[t] = W[t] + h[l] * X[index]
            V[t] = V[t] + g[l] * X[index]
    return (W, V)

def get_X(h, g, j, W, V):
    """
    Level j of inverse pyramid algorithm.
    Take W_j and V_j and return V_(j-1)
    
    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the MODWT wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the MODWT scaling filter
        type j = integer
        j = Current level of the pyramid algorithm
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
    X = np.zeros(N)
    L = np.shape(h)[0]
    for t in range(0, N):
        for l in range(0, L):
            index = (t + (2 ** (j - 1)) * l) % N
            X[t] = X[t] + h[l] * W[index] + g[l] * V[index]
    return X

def pyramid(X, name, J):
    """
    Compute the MODDWT of X up to level J
    
    Input:
        type X = 1D numpy array
        X = Time series
        type name = string
        name = Name of the MODWT wavelet filter
        type J = integer
        J = Level of partial MODWT
    Output:
        type W = list of 1D numpy arrays (length J)
        W = List of vectors of MODWT wavelet coefficients
        type Vj = 1D numpy array
        Vj = Vector of MODWT scaling coefficients at level J
    """
    assert (type(J) == int), \
        'Level of DWT must be an integer'
    assert (J >= 1), \
        'Level of DWT must be higher or equal to 1'
    g = get_scaling(name)
    h = get_wavelet(g)
    Vj = X
    W = []
    for j in range(1, (J + 1)):
        (Wj, Vj) = get_WV(h, g, j, Vj)
        W.append(Wj)
    return (W, Vj)

def inv_pyramid(W, Vj, name, J):
    """
    Compute the inverse MODWT of W up to level J
    
    Input:
        type W = list of 1D numpy arrays (length J)
        W = List of vectors of MODWT wavelet coefficients
        type Vj = 1D numpy array
        Vj = Vector of MODWT scaling coefficients at level J
        type name = string
        name = Name of the MODWT wavelet filter
        type J = integer
        J = Level of partial MODWT
    Output:
        type X = 1D numpy array
        X = Original time series
    """
    assert (type(J) == int), \
        'Level of DWT must be an integer'
    assert (J >= 1), \
        'Level of DWT must be higher or equal to 1'
    g = get_scaling(name)
    h = get_wavelet(g)
    for j in range(J, 0, -1):
        Vj = get_X(h, g, j, W[j - 1], Vj)
    X = Vj
    return X

def get_DS(X, W, name, J):
    """
    Compute the details and the smooths of the time series
    using the MODWT coefficients

    Input:
        type X = 1D numpy array
        X =  Time series
        type W = list of 1D numpy arrays (length J)
        W = List of vectors of MODWT wavelet coefficients
        type name = string
        name = Name of the MODWT wavelet filter
        type J = integer
        J = Level of partial MODWT
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
    N = np.shape(X)[0]
    # Compute details
    D = []
    for j in range(1, J + 1):
        Wj = []
        if (j > 1):
            for k in range(1, j):
                Wj.append(np.zeros(N))
        Wj.append(W[j - 1])
        Vj = np.zeros(N)
        Dj = inv_pyramid(Wj, Vj, name, j)
        D.append(Dj)
    # Compute smooths
    S = [X]
    for j in range(0, J):
        Sj = S[-1] - D[j]
        S.append(Sj)
    return (D, S)
