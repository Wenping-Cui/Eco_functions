import numpy as np
import pickle
import os


def appendDFToCSV_void(df, csvFilePath, sep=","):
    """Append a DataFrame to a CSV file, writing the header only on first write."""
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=True, sep=sep)
    else:
        df.to_csv(csvFilePath, mode='a', index=True, sep=sep, header=False)


def load_parameters(filename):
    """Load simulation parameters from a pickle file.

    Returns a tuple (message, params) where params is None if the file is missing.
    """
    if os.path.exists(filename):
        sim_par = pickle.load(open(filename, "rb"))
        return 'File exists and load existed parameter', sim_par
    else:
        return 'File does not exist', None


def save_parameters(sim, filename):
    """Save simulation parameters to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(sim, f)
    return 'File is saved'


def KL_lams(lams, sigc, phiN, phiR, M, Nbins=100):
    """KL divergence between the empirical eigenvalue distribution and the
    Marchenko-Pastur prediction.

    Parameters
    ----------
    lams : array-like
        Empirical eigenvalues.
    sigc : float
        Standard deviation of the consumption matrix entries (scaled).
    phiN : float
        Fraction of surviving species.
    phiR : float
        Fraction of surviving resources.
    M : int
        Number of resources.
    Nbins : int
        Number of histogram bins.
    """
    s = sigc * np.sqrt(M)
    c = phiN / phiR
    a = (s**2) * (1 - np.sqrt(c))**2
    b = (s**2) * (1 + np.sqrt(c))**2
    bins = np.linspace(a, b, Nbins)
    p, bins = np.histogram(lams, bins=bins, density=True)
    bins = np.linspace(a, b, Nbins - 1)
    q = MP_density(bins, a, b, c, s)
    q = np.nan_to_num(q)
    return KL(p, q)


def MP_density(bins, a, b, c, s):
    """Marchenko-Pastur spectral density evaluated at bin centres."""
    return (1. / (2 * np.pi * bins * c * s**2)) * np.sqrt((b - bins) * (bins - a))


def KL(P, Q, epsilon=1e-5):
    """KL divergence KL(P || Q), regularised by epsilon to avoid log(0)."""
    P = P + epsilon
    Q = Q + epsilon
    return np.sum(P * np.log(P / Q))
