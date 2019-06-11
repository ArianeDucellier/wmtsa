"""
This module contains functions to carry out the best basis algorithm
"""

import numpy as np

from math import log

import DWPT

def compute_m(W, typecost, normX=1, delta=0.5, p=1):
    """
    Compute the cost functional of Wj,n,t

    Input:
        type W = float
        W = Wj,n,t
        type typecost = string
        typecost = 'entropy' for the -l2 log(l2) norm
                   'thresh' for the threshold functional
                   'lp' for the lp norm (see WMTSA p 223)
        type normX = float
        normX = Norm of the time series 
        type delta = float
        delta = Value of the threshold for the threshold functional (> 0)
        type p = float
        p = Power for the lp norm (0 < p < 2)
    Output:
        type m = float
        m = Cost function of Wj,n,t
    """
    if (typecost == 'entropy'):
        assert (normX > 0), \
            'The norm of the time series must be strictly positive'
        Wbar = W / normX
        if (W == 0.0):
            m = 0.0
        else:
            m = - (Wbar ** 2.0) * log(Wbar ** 2.0)
    elif (typecost == 'thresh'):
        assert (delta > 0), \
            'The threshold must be strictly positive'
        if (abs(W) > delta):
            m = 1.0
        else:
            m = 0.0
    elif (typecost == 'lp'):
        assert ((p > 0) and (p < 2)), \
            'We must have 0 < p < 2 for the lp norm'
        m = (abs(W)) ** p
    else:
        raise ValueError('The cost functional must be entropy, thresh or lp')
    return m

def compute_M(W, typecost, normX=1, delta=0.5, p=1):
    """
    Compute the cost functional of Wj,n

    Input:
        type W = float
        W = Wj,n
        type typecost = string
        typecost = 'entropy' for the -l2 log(l2) norm
                   'thresh' for the threshold functional
                   'lp' for the lp norm (see WMTSA p 223)
        type normX = float
        normX = Norm of the time series 
        type delta = float
        delta = Value of the threshold for the threshold functional (> 0)
        type p = float
        p = Power for the lp norm (0 < p < 2)
    Output:
        type M = float
        M = Cost function of Wj,n
    """
    M = 0.0
    for i in range(0, len(W)):
        m = compute_m(W[i], typecost, normX, delta, p)
        M = M + m
    return M

def bestbasis(W, X, typecost, delta=0.5, p=1):
    """
    Compute the best basis algorithm from WMTSA (pp 225-226)

    Input:
        type W = list of J+1 1D numpy arrays
        W = Vectors of DWPT coefficients at levels 0, ... , J
            (output of DWPT.get_DWPT)
        type X = 1D numpy array
        X = Time series
        type typecost = string
        typecost = 'entropy' for the -l2 log(l2) norm
                   'thresh' for the threshold functional
                   'lp' for the lp norm (see WMTSA p 223)
        type delta = float
        delta = Value of the threshold for the threshold functional (> 0)
        type p = float
        p = Power for the lp norm (0 < p < 2)
    Output:
        type base = list of lists of integers
        base = Indices of the vectors for the best basis
               Value = 1 if it is in the best basis, otherwise value = 0
    """
    normX = np.sqrt(np.sum(np.square(X)))
    N = np.shape(X)[0]
    J = len(W) - 1
    cost = []
    base = []
    for j in range(0, J + 1):
        Wj = W[j]
        Nj = int(N / (2 ** j))
        costj = []
        for n in range(0, 2 ** j):
            Wjn = Wj[int(n * Nj) : int((n + 1) * Nj)]
            M = compute_M(Wjn, typecost, normX, delta, p)
            costj.append(M)
        if (j == J):
            basej = [1] * int(2 ** j)
        else:
            basej = [0] * int(2 ** j)
        cost.append(costj)
        base.append(basej)
    for j in range(J, 0, -1):
        for n in range(0, int(2 ** (j - 1))):
            if (cost[j - 1][n] <= cost[j][int(2 * n)] + \
                                  cost[j][int(2 * n + 1)]):
                base[j - 1][n] = 1
                base[j][int(2 * n)] = 0
                base[j][int(2 * n + 1)] = 0
            else:
                cost[j - 1][n] = cost[j][int(2 * n)] + cost[j][int(2 * n + 1)]
    return (base, cost)

def decomposition(X, name, J, typecost, delta=0.5, p=1):
    """
    Get the decomposition of X into the best wavelet vector basis

    Input:
        type X = 1D numpy array
        X = Time series which length is a multiple of 2**J
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial DWT
        type typecost = string
        typecost = 'entropy' for the -l2 log(l2) norm
                   'thresh' for the threshold functional
                   'lp' for the lp norm (see WMTSA p 223)
        type delta = float
        delta = Value of the threshold for the threshold functional (> 0)
        type p = float
        p = Power for the lp norm (0 < p < 2)
    Output:
        type w = list of scalars
        w = Weights
        type b = list of vectors of length N
        b = Basis vectors
    """
    assert (type(J) == int), \
        'Level of DWT must be an integer'
    assert (J >= 1), \
        'Level of DWT must be higher or equal to 1'
    N = np.shape(X)[0]
    assert (N % (2 ** J) == 0), \
        'Length of time series is not a multiple of 2**J'
    W = DWPT.get_DWPT(X, name, J)
    c = DWPT.compute_c(J)
    (base, cost) = bestbasis(W, X, typecost, delta, p)
    w = []
    b = []
    if (base[0][0] == 1):
        w.append(1.0)
        b.append(X)
    else:
        for j in range(1, len(base)): 
            for n in range(0, int(2 ** j)):
                if (base[j][n] == 1):
                    weights = W[j][int(n * N / (2 ** j)) : \
                                   int((n + 1) * N / (2 ** j))]
                    cjn = c[j - 1][n]
                    vectors = DWPT.compute_basisvector(cjn, name, N)
                    for l in range(0, len(weights)):
                        w.append(weights[l])
                    for l in range(0, np.shape(vectors)[1]):
                        b.append(vectors[:, l])
    return (w, b)
