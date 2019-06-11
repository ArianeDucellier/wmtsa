"""
This module contains functions to compute the MODWPT of a time series
"""

import matplotlib.pyplot as plt
import numpy as np

import DWPT, MODWT

def get_Wjn(h, g, j, n, Wjm1n):
    """
    Compute the MODWPT coefficients Wj,2n,t and Wj,2n+1,t at level j
    from the MODWPT coefficients Wj-1,n,t at level j - 1

    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the MODWPT wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the MODWPT scaling filter
        type j = integer
        j = Current level of the MODWPT decomposition
        type n = integer
        n = Index of the MODWPT vector Wj-1,n at level j - 1
        type Wjm1n = 1D numpy array
        Wjm1n = MODWPT coefficients Wj-1,n,t at level j - 1
    Output:
        type Wjn1 = 1D numpy array
        Wjn1 = MODWPT coefficients Wj,2n,t at level j
        type Wjn2 = 1D numpy array
        Wjn2 = MODWPT coefficients Wj,2n+1,t at level j
    """
    assert (np.shape(h)[0] == np.shape(g)[0]), \
        'Wavelet and scaling filters have different lengths'
    N = len(Wjm1n)
    assert ((n >= 0) and (n <= 2 ** (j - 1) - 1)), \
        'Index n must be >= 0 and <= 2 ** (j - 1) - 1'
    Wjn1 = np.zeros(N)
    Wjn2 = np.zeros(N)
    L = np.shape(h)[0]
    if (n % 2 == 0):
        an = g
        bn = h
    else:
        an = h
        bn = g
    for t in range(0, N):
        for l in range(0, L):
            index = int((t - (2 ** (j - 1)) * l) % N)
            Wjn1[t] = Wjn1[t] + an[l] * Wjm1n[index]
            Wjn2[t] = Wjn2[t] + bn[l] * Wjm1n[index]
    return (Wjn1, Wjn2)

def get_Wjm1n(h, g, j, n, Wjn1, Wjn2):
    """
    Compute the MODWPT coefficients Wj-1,n,t at level j - 1
    from the MODWPT coefficients Wj,2n,t and Wj,2n+1,t at level j

    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the MODWPT wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the MODWPT scaling filter
        type j = integer
        j = Current level of the MODWPT decomposition
        type n = integer
        n = Index of the MODWPT vector Wj-1,n at level j - 1
        type Wjn1 = 1D numpy array
        Wjn1 = MODWPT coefficients Wj,2n,t at level j
        type Wjn2 = 1D numpy array
        Wjn2 = MODWPT coefficients Wj,2n+1,t at level j
    Output:
        type Wjm1n = 1D numpy array
        Wjm1n = MODWPT coefficients Wj-1,n,t at level j - 1        
    """
    assert (np.shape(h)[0] == np.shape(g)[0]), \
        'Wavelet and scaling filters have different lengths'
    assert (len(Wjn1) == len(Wjn2)), \
        'MODWPT coefficients Wj,2n,t and Wj,2n+1,t must have the same length'
    N = len(Wjn1)
    assert ((n >= 0) and (n <= 2 ** (j - 1) - 1)), \
        'Index n must be >= 0 and <= 2 ** (j - 1) - 1'
    Wjm1n = np.zeros(N)
    L = np.shape(h)[0]
    if (n % 2 == 0):
        an = g
        bn = h
    else:
        an = h
        bn = g
    for t in range(0, N):
        for l in range(0, L):
            index = int((t + (2 ** (j - 1)) * l) % N)
            Wjm1n[t] = Wjm1n[t] + an[l] * Wjn1[index] + bn[l] * Wjn2[index]
    return Wjm1n

def get_Wj(h, g, j, Wjm1):
    """
    Compute the MODWPT coefficients Wj at level j
    from the MODWPT coefficients Wj-1 at level j - 1

    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the MODWPT wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the MODWPT scaling filter
        type j = integer
        j = Current level of the MODWPT decomposition
        type Wjm1 = list of n = 2 ** (j - 1) 1D numpy arrays
        Wjm1 = MODWPT coefficients Wj-1 at level j - 1
    Output:
        type Wj = list of n = 2 ** j 1D numpy arrays
        Wj = MODWPT coefficients Wj at level j
    """
    assert (np.shape(h)[0] == np.shape(g)[0]), \
        'Wavelet and scaling filters have different lengths'
    Wj = []
    for n in range(0, 2 ** (j - 1)):
        Wjm1n = Wjm1[n]
        (Wjn1, Wjn2) = get_Wjn(h, g, j, n, Wjm1n)
        Wj.append(Wjn1)
        Wj.append(Wjn2)
    return Wj

def get_Wjm1(h, g, j, Wj):
    """
    Compute the MODWPT coefficients Wj-1 at level j - 1
    from the MODWPT coefficients Wj at level j

    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the MODWPT wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the MODWPT scaling filter
        type j = integer
        j = Current level of the MODWPT decomposition
        type Wj = list of n = 2 ** j 1D numpy arrays
        Wj = MODWPT coefficients Wj at level j
    Output:
        type Wjm1 = list of n = 2 ** (j - 1) 1D numpy arrays
        Wjm1 = MODWPT coefficients Wj-1 at level j - 1
    """
    assert (np.shape(h)[0] == np.shape(g)[0]), \
        'Wavelet and scaling filters have different lengths'
    Wjm1 = []
    for n in range(0, 2 ** (j - 1)):
        Wjn1 = Wj[2 * n]
        Wjn2 = Wj[2 * n + 1]
        Wjm1n = get_Wjm1n(h, g, j, n, Wjn1, Wjn2)
        Wjm1.append(Wjm1n)
    return Wjm1

def get_MODWPT(X, name, J):
    """
    Compute the MODWPT of X up to level J
    
    Input:
        type X = 1D numpy array
        X = Time series
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial MODWPT
    Output:
        type W = list of J+1 lists of 1D numpy arrays
        W = Vectors of MODWPT coefficients at levels 0, ... , J
    """
    assert (type(J) == int), \
        'Level of MODWPT must be an integer'
    assert (J >= 1), \
        'Level of MODWPT must be higher or equal to 1'
    g = MODWT.get_scaling(name)
    h = MODWT.get_wavelet(g)
    W = [[X]]
    for j in range(1, J + 1):
        Wjm1 = W[-1]
        Wj = get_Wj(h, g, j, Wjm1)
        W.append(Wj)
    return W

def inv_MODWPT(WJ, name, J):
    """
    Compute the inverse MODWPT of W at level J
    
    Input:
        type WJ = list of n = 2 ** J 1D numpy arrays
        WJ = Vectors of MODWPT coefficients at level J      
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial MODWPT
    Output:
        type X = 1D numpy array
        X = Time series
    """
    assert (type(J) == int), \
        'Level of MODWPT must be an integer'
    assert (J >= 1), \
        'Level of MODWPT must be higher or equal to 1'
    assert (len(WJ) == 2 ** J), \
        'MODWPT decomposition at level {} must contain {} vectors'. \
        format(J, int(2 ** J))
    g = MODWT.get_scaling(name)
    h = MODWT.get_wavelet(g)
    Wj = WJ
    for j in range(J, 0, -1):
        Wjm1 = get_Wjm1(h, g, j, Wj)
        Wj = Wjm1
    X = np.reshape(Wj, np.shape(Wj)[1])
    return X

def get_Dj(Wj, name, j):
    """
    Compute the details at level j using the wavelet coefficients

    Input:
        type Wj = list of n = 2 ** j 1D numpy arrays
        Wj = Vectors of MODWPT coefficients at level j
        type name = string
        name = Name of the wavelet filter
        type j = integer
        j = Level of partial MODWPT
    Output:
        type Dj = list of n = 2 ** j 1D numpy arrays
        Dj = Vectors of MODWPT details at level j
    """
    assert (type(j) == int), \
        'Level of MODWPT must be an integer'
    assert (j >= 1), \
        'Level of MODWPT must be higher or equal to 1'
    assert (len(Wj) == 2 ** j), \
        'MODWPT decomposition at level {} must contain {} vectors'. \
        format(j, int(2 ** j))
    Dj = []
    for n in range(0, 2 ** j):
        W = []
        for m in range(0, 2 ** j):
            if (m == n):
                W.append(Wj[n])
            else:
                W.append(np.zeros(len(Wj[n])))
        Dn = inv_MODWPT(W, name, j)
        Dj.append(Dn)
    return Dj
