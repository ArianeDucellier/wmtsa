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

if __name__ == '__main__':

    # Test 1
    def test1():
        """
        Reproduce plot of Figure 183 from WMTSA

        Input:
            None
        Output:
            None
        """
        X = np.loadtxt('../tests/data/heart.dat')
        N = np.shape(X)[0]
        L = 8
        (W, V) = pyramid(X, 'LA8', 6)
        (nuH, nuG) = DWT.get_nu('LA8', 6)
        plt.figure(1, figsize=(15, 32))
        plt.subplot2grid((8, 1), (7, 0))
        dt = 1.0 / 180.0
        t = dt * np.arange(0, N)
        plt.plot(t, X, 'k', label='X')
        plt.xlim([np.min(t), np.max(t)])
        plt.legend(loc=1)
        for j in range(0, 6):
            plt.subplot2grid((8, 1), (6 - j, 0))
            plt.plot(t, np.roll(W[j], nuH[j]), 'k', \
                label='T' + str(nuH[j]) + 'W' + str(j + 1))
            Lj = (2 ** (j + 1) - 1) * (L - 1) + 1
            plt.axvline(dt * (Lj - 2 - abs(nuH[j])), linewidth=1, color='red')
            plt.axvline(dt * (N - abs(nuH[j])), linewidth=1, color='red')
            plt.xlim([np.min(t), np.max(t)])
            plt.legend(loc=1)
        plt.subplot2grid((8, 1), (0, 0))
        plt.plot(t, np.roll(V, nuG[5]), 'k', \
            label='T' + str(nuG[5]) + 'V6')
        Lj = (2 ** 6 - 1) * (L - 1) + 1
        plt.axvline(dt * (Lj - 2 - abs(nuG[j])), linewidth=1, color='red')
        plt.axvline(dt * (N - abs(nuG[j])), linewidth=1, color='red')
        plt.xlim([np.min(t), np.max(t)])
        plt.legend(loc=1)
        plt.title('MODWT of ECG time series')
        plt.savefig('../tests/MODWT/ECG_WV.eps', format='eps')
        plt.close(1)

    # Compute MODWT coefficients of the ECG time series from WMTSA
    # See Figure 183 in WMTSA
    test1()

    # Test 2
    def test2():
        """
        Reproduce plot of Figure 184 from WMTSA

        Input:
            None
        Output:
            None
        """
        X = np.loadtxt('../tests/data/heart.dat')
        N = np.shape(X)[0]
        L = 8
        (W, V) = pyramid(X, 'LA8', 6)
        (D, S) = get_DS(X, W, 'LA8', 6)
        plt.figure(1, figsize=(15, 32))
        plt.subplot2grid((8, 1), (7, 0))
        dt = 1.0 / 180.0
        t = dt * np.arange(0, N)
        plt.plot(t, X, 'k', label='X')
        plt.xlim([np.min(t), np.max(t)])
        plt.legend(loc=1)
        for j in range(0, 6):
            plt.subplot2grid((8, 1), (6 - j, 0))
            plt.plot(t, D[j], 'k', label='D' + str(j + 1))
            Lj = (2 ** (j + 1) - 1) * (L - 1) + 1
            plt.axvline(dt * (Lj - 2), linewidth=1, color='red')
            plt.axvline(dt * (N - Lj + 1), linewidth=1, color='red')
            plt.xlim([np.min(t), np.max(t)])
            plt.legend(loc=1)
        plt.subplot2grid((8, 1), (0, 0))
        plt.plot(t, S[6], 'k', label='S6')
        Lj = (2 ** 6 - 1) * (L - 1) + 1
        plt.axvline(dt * (Lj - 2), linewidth=1, color='red')
        plt.axvline(dt * (N - Lj + 1), linewidth=1, color='red')
        plt.xlim([np.min(t), np.max(t)])
        plt.legend(loc=1)
        plt.title('MODWT of ECG time series')
        plt.savefig('../tests/MODWT/ECG_DS.eps', format='eps')
        plt.close(1)

    # Compute MODWT MRA of the ECG time series from WMTSA
    # See Figure 184 in WMTSA
    test2()

    # Test 3
    def test3(xmin, xmax, name_output):
        """
        Reproduce plots of Figures 186 and 187 from WMTSA

        Input:
            type xmin = float
            xmin = Time where to begin the plot
            type xmax = float
            xmax = Time where to and the plot
            type name_output = string
            name_output = Name of image file containing the plot
        Output:
            None
        """
        X = np.loadtxt('../tests/data/subtidal.dat')
        N = np.shape(X)[0]
        L = 8
        (W, V) = pyramid(X, 'LA8', 7)
        (D, S) = get_DS(X, W, 'LA8', 7)
        plt.figure(1, figsize=(15, 36))
        plt.subplot2grid((9, 1), (8, 0))
        t1980 = 1980.0 + (0.5 / 366.0) * np.arange(10, 732)
        t1981 = 1981.0 + (0.5 / 365.0) * np.arange(0, 730)
        t1982 = 1982.0 + (0.5 / 365.0) * np.arange(0, 730)
        t1983 = 1983.0 + (0.5 / 365.0) * np.arange(0, 730)
        t1984 = 1984.0 + (0.5 / 366.0) * np.arange(0, 732)
        t1985 = 1985.0 + (0.5 / 365.0) * np.arange(0, 730)
        t1986 = 1986.0 + (0.5 / 365.0) * np.arange(0, 730)
        t1987 = 1987.0 + (0.5 / 365.0) * np.arange(0, 730)
        t1988 = 1988.0 + (0.5 / 366.0) * np.arange(0, 732)
        t1989 = 1989.0 + (0.5 / 365.0) * np.arange(0, 730)
        t1990 = 1990.0 + (0.5 / 365.0) * np.arange(0, 730)
        t1991 = 1991.0 + (0.5 / 365.0) * np.arange(0, 720)
        t = np.concatenate((t1980, t1981, t1982, t1983, t1984, t1985, \
            t1986, t1987, t1988, t1989, t1990, t1991))
        plt.plot(t, X, 'k', label='X')
        plt.xlim([xmin, xmax])
        plt.legend(loc=1)
        for j in range(0, 7):
            plt.subplot2grid((9, 1), (7 - j, 0))
            plt.plot(t, D[j], 'k', label='D' + str(j + 1))
            Lj = int((2 ** (j + 1) - 1) * (L - 1) + 1)
            plt.axvline(t[Lj - 2], linewidth=1, color='red')
            plt.axvline(t[N - Lj + 1], linewidth=1, color='red')
            plt.xlim([xmin, xmax])
            plt.legend(loc=1)
        plt.subplot2grid((9, 1), (0, 0))
        plt.plot(t, S[7], 'k', label='S7')
        Lj = (2 ** 7 - 1) * (L - 1) + 1
        plt.axvline(t[Lj - 2], linewidth=1, color='red')
        plt.axvline(t[N - Lj + 1], linewidth=1, color='red')
        plt.xlim([xmin, xmax])
        plt.legend(loc=1)
        plt.title('MODWT of Crescent City subtidal variations')
        plt.savefig('../tests/MODWT/' + name_output + '.eps', \
            format='eps')
        plt.close(1)

    # Compute MODWT MRA of the subtidal time series from WMTSA
    # See Figures 186 and 187 in WMTSA
    test3(1980.0 + 5.0 / 366.0, 1991 + 359.5 / 365.0, 'subtidal_1980-1991')
    test3(1985.0, 1987.0, 'subtidal_1985-1986')

    def test4():
        """
        Reproduce plot of Figure 192 from WMTSA

        Input:
            None
        Output:
            None
        """
        X = np.loadtxt('../tests/data/Nile.dat')
        N = np.shape(X)[0]
        L = 2
        (W, V) = pyramid(X, 'Haar', 4)
        (D, S) = get_DS(X, W, 'Haar', 4)
        plt.figure(1, figsize=(15, 24))
        plt.subplot2grid((6, 1), (5, 0))
        t = np.arange(622, 1285, 1)
        plt.plot(t, X, 'k', label='X')
        plt.xlim([np.min(t), np.max(t)])
        plt.legend(loc=1)
        for j in range(0, 4):
            plt.subplot2grid((6, 1), (4 - j, 0))
            plt.plot(t, D[j], 'k', label='D' + str(j + 1))
            Lj = int((2 ** (j + 1) - 1) * (L - 1) + 1)
            plt.axvline(t[Lj - 2], linewidth=1, color='red')
            plt.axvline(t[N - Lj + 1], linewidth=1, color='red')
            plt.xlim([np.min(t), np.max(t)])
            plt.legend(loc=1)
        plt.subplot2grid((6, 1), (0, 0))
        plt.plot(t, S[4], 'k', label='S4')
        Lj = (2 ** 4 - 1) * (L - 1) + 1
        plt.axvline(t[Lj - 2], linewidth=1, color='red')
        plt.axvline(t[N - Lj + 1], linewidth=1, color='red')
        plt.xlim([np.min(t), np.max(t)])
        plt.legend(loc=1)
        plt.title('MODWT of Nile river minima')
        plt.savefig('../tests/MODWT/Nile.eps', format='eps')
        plt.close(1)

    # Compute MODWT MRA of the Nile river time series from WMTSA
    # See Figure 192 in WMTSA
    test4()

    # Test 5
    def test5():
        """
        Reproduce plot of Figure 194 from WMTSA

        Input:
            None
        Output:
            None
        """
        X = np.loadtxt('../tests/data/msp.dat')
        N = np.shape(X)[0]
        L = 8
        (W, V) = pyramid(X, 'LA8', 6)
        (D, S) = get_DS(X, W, 'LA8', 6)
        plt.figure(1, figsize=(15, 32))
        plt.subplot2grid((8, 1), (7, 0))
        dt = 0.1
        t = 350.0 + dt * np.arange(0, N)
        plt.plot(t, X, 'k', label='X')
        plt.xlabel('depth (meters)')
        plt.xlim([np.min(t), np.max(t)])
        plt.legend(loc=1)
        for j in range(0, 6):
            plt.subplot2grid((8, 1), (6 - j, 0))
            plt.plot(t, D[j], 'k', label='D' + str(j + 1))
            Lj = (2 ** (j + 1) - 1) * (L - 1) + 1
            plt.axvline(t[Lj - 2], linewidth=1, color='red')
            plt.axvline(t[N - Lj + 1], linewidth=1, color='red')
            plt.xlim([np.min(t), np.max(t)])
            plt.legend(loc=1)
        plt.subplot2grid((8, 1), (0, 0))
        plt.plot(t, S[6], 'k', label='S6')
        Lj = (2 ** 6 - 1) * (L - 1) + 1
        plt.axvline(t[Lj - 2], linewidth=1, color='red')
        plt.axvline(t[N - Lj + 1], linewidth=1, color='red')
        plt.xlim([np.min(t), np.max(t)])
        plt.legend(loc=1)
        plt.title('MODWT of ocean shear time series')
        plt.savefig('../tests/MODWT/ocean_shear.eps', format='eps')
        plt.close(1)

    # Compute MODWT MRA of the ocean shear time series from WMTSA
    # See Figure 194 in WMTSA
    test5()
