import numpy as np
from scipy.linalg import block_diag
import math as mt


def Consum_matrix(c_m, c_sigma, q_c, M, F, flag_family):
    """Generate a single species' consumption rate vector with family preference.

    Species in family ``flag_family`` (every F-th resource) consume at a higher
    rate scaled by ``q_c``; all others are scaled by ``1 - q_c``.

    Parameters
    ----------
    c_m : float
        Mean consumption rate.
    c_sigma : float
        Standard deviation of consumption rate.
    q_c : float
        Preference strength (0 = generalist, 1 = specialist).
    M : int
        Number of resources.
    F : int
        Number of resource families (period of the preference pattern).
    flag_family : int
        Index of the preferred resource family (0 to F-1).

    Returns
    -------
    c : ndarray, shape (M,)
        Consumption rates, normalised so their sum equals M * c_m.
    """
    c = np.zeros(M)
    for j in range(M):
        if j % F == flag_family:
            c[j] = np.abs(np.random.normal(c_m, c_sigma) * q_c)
        else:
            c[j] = np.abs(np.random.normal(c_m, c_sigma) * (1 - q_c))
    c = c / np.sum(c) * M * c_m
    # Ensure first resource has a background rate
    c[0] = np.abs(np.random.normal(c_m, c_sigma))
    return c


def block(n, m):
    """Build an (n*m) × (n*m) block-diagonal matrix with all-ones blocks.

    Parameters
    ----------
    n : int
        Number of blocks.
    m : int
        Size of each square block.

    Returns
    -------
    C : ndarray, shape (n*m, n*m)
    """
    a = np.full((m, m), 1)
    return block_diag(*([a] * n))


def circ(n, r):
    """Build an n×n circulant matrix with Gaussian-decay entries.

    Entry (i, j) decays as exp(-d^2 / (2 r^2)) where d = min(|i-j|, n-|i-j|)
    is the circular distance.

    Parameters
    ----------
    n : int
        Matrix dimension.
    r : float
        Decay radius (standard deviation of the Gaussian kernel).

    Returns
    -------
    C : ndarray, shape (n, n)
    """
    cc = np.array([mt.exp(-min(i, abs(n - i))**2 / (2 * r**2)) for i in range(n)])
    C = cc.copy()
    for _ in range(n - 1):
        cc = np.append(cc[-1:], cc[:-1])
        C = np.vstack((C, cc))
    return C


def binomial(p, S, M):
    """Generate a random binary S×M consumption matrix with entry probability p.

    Parameters
    ----------
    p : float
        Probability that each entry is 1.
    S : int
        Number of species (rows).
    M : int
        Number of resources (columns).

    Returns
    -------
    c : ndarray, shape (S, M)
    """
    c = np.zeros((S, M))
    for i in range(S):
        for j in range(M):
            if np.random.rand() < p:
                c[i, j] = 1.0
    return c
