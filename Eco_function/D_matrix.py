import numpy as np

##########################################################################
# metabolic_flag = "random" | "fixed" | "tiled" | "one-step"
##########################################################################


def Make_D_matrices(S, M, nu, p, q, energies, flag):
    """Build per-species metabolic (crossfeeding) D matrices.

    Each D[a, b] encodes the fraction of resource b that is converted to
    resource a upon consumption (metabolic byproduct routing).

    Parameters
    ----------
    S : int
        Number of species.
    M : int
        Number of resources.
    nu : int
        Minimum trophic depth; resource indices below ``nu`` are always zero.
    p : float
        Total leakage fraction (rows of D sum to p after normalisation).
    q : float
        Probability that a given row has any nonzero entries.
    energies : array-like, shape (M,)
        Energy content of each resource (used for energy-conserving normalisation).
    flag : tuple (metabolic_flag, community_flag)
        metabolic_flag – one of "random", "fixed", "tiled", "one-step".
        community_flag – "A", "B", or "" to select community-specific routing.

    Returns
    -------
    list of ndarray, each of shape (M, M)
        One D matrix per species (or per resource pair for "one-step").
    """
    metabolic_flag = flag[0]
    community_flag = flag[1]
    nu = np.minimum(M - 1, nu)

    # Build per-species trophic depth array
    if metabolic_flag == "random":
        nu_array = np.append(np.array([nu]), np.random.randint(nu, M - 1, size=S - 1))
    elif metabolic_flag == "fixed":
        nu_array = nu * np.ones(S)
    elif metabolic_flag == "tiled":
        nu_array = np.mod(np.arange(nu, S + 1) - nu, M - nu) + nu
    elif metabolic_flag == "one-step":
        nu_array = nu * np.ones(S)
    else:
        raise ValueError(f"Unknown metabolic_flag: '{metabolic_flag}'")

    ecosystem = []
    for j in range(S):
        nu_j = int(nu_array[j])
        D_matrix = np.zeros((M, M))

        # Randomly activate rows based on community type
        for a in range(M):
            if np.random.rand() <= q:
                D_matrix[a, :] = 1.0
                if np.random.rand() < 0.95:
                    if community_flag == 'A' and a % 2 == 1:
                        D_matrix[a, :] = 0.0
                    if community_flag == 'B' and a % 2 == 0:
                        D_matrix[a, :] = 0.0

        # Enforce community-specific anchor rows
        if community_flag == 'A':
            D_matrix[M - 2, :] = 1.0
            D_matrix[M - 1, :] = 0
        if community_flag == 'B':
            D_matrix[M - 1, :] = 1.0
            D_matrix[M - 2, :] = 0

        # Zero out entries that violate trophic ordering
        for a in range(M):
            for b in range(M):
                if a <= b or b < nu_j:
                    D_matrix[a, b] = 0

        # Normalise nonzero columns so that total leakage equals p,
        # rescaled by the energy ratio energies[i] / energies[indice]
        for i in range(M):
            D_vec = np.copy(D_matrix[:, i])
            nonzero_idx = np.nonzero(D_vec)[0]
            if len(nonzero_idx) > 0:
                r = np.random.rand(len(nonzero_idx))
                r = r / np.sum(r) * p
                for k, idx in enumerate(nonzero_idx):
                    D_vec[idx] = r[k] / energies[idx] * energies[i]
            D_matrix[:, i] = D_vec

        D_matrix[D_matrix < 1e-5] = 0
        ecosystem.append(D_matrix)

    # "one-step": each resource converts only to the next resource
    if metabolic_flag == "one-step":
        ecosystem1 = []
        for j in range(M - 1):
            D = np.zeros((M, M))
            D[j + 1, j] = 1
            ecosystem1.append(D)
        return ecosystem1

    return ecosystem
