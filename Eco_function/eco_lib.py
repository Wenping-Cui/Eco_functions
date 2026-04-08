import numpy as np
from scipy.integrate import odeint

##########################################################################


class Ecology_simulation(object):
    """ODE-based consumer-resource model for a single community realisation.

    Parameters are passed as an ordered list ``par`` built by
    ``CRM_Cavity_simulation.initialize_random_variable()``.

    Dynamics flags (set after construction, before calling ``simulation``):
        flag_renew      – True  → renewable resources (linear or nonlinear)
        flag_linear     – True  → Type-I (linear) resource renewal
        flag_nonvanish  – True  → constant resource input (K not depleted)
        flag_crossfeeding – True → metabolic byproduct recycling via matrix D
        gamma / K_sat   – parameters for Type-II functional response
        e               – metabolic efficiency for crossfeeding
    """

    def __init__(self, par):
        self.flag_crossfeeding = par[0]
        self.M = par[1]
        self.S = par[2]
        self.R_ini = par[3]
        self.N_ini = par[4]
        self.t0, self.t1, self.Nt = par[5]
        self.C = par[6]
        self.energies = par[7]
        self.tau_inv = par[8]
        self.costs = par[9]
        self.growth = par[10]
        if self.flag_crossfeeding:
            self.D = par[11]
            self.non_zero_resource = par[12]
            self.resource_amount = par[13]
            self.K = self.resource_amount
            self.power_max = np.dot(np.dot(self.resource_amount, self.energies), self.tau_inv)
        else:
            self.K = par[11]
            self.power_max = np.dot(self.K * self.tau_inv, self.energies)
        self.costs_power = 0
        self.eff = 0
        self.R_f = np.zeros(self.M)
        self.N_f = np.zeros(self.S)
        self.survive = 0
        self.flag_renew = False
        self.flag_linear = False
        self.gamma = 1
        self.K_sat = 1
        self.flag_nonvanish = False
        self.e = 0

    ##########################################################################
    # Public interface
    ##########################################################################

    def simulation(self):
        """Integrate the ODEs and store final resource/species abundances.

        Returns
        -------
        R_traj : ndarray, shape (Nt, M)
        N_traj : ndarray, shape (Nt, S)
        """
        Y_ini = np.concatenate((self.R_ini, self.N_ini))
        T = np.linspace(self.t0, self.t1, num=self.Nt)
        self.costs = np.asarray(self.costs)
        self.C = np.asarray(self.C)

        if self.flag_crossfeeding:
            self.D = np.asarray(self.D)
            self.R0 = self.resource_amount
            self.dynamics = self.dynamics_nonrenewable_typeI_crossfeeding_on
            par = [self.M, self.S, self.R0, self.energies, self.tau_inv,
                   self.costs, self.growth, self.C, self.D, self.e]
        else:
            par = [self.M, self.S, self.K, self.energies, self.tau_inv,
                   self.costs, self.growth, self.C]
            if self.flag_renew:
                self.dynamics = (self.get_vector_field_crossfeeding_off
                                 if self.flag_linear
                                 else self.get_vector_field_crossfeeding_off_nonlinear)
            else:
                self.dynamics = self.get_vector_field_crossfeeding_off_nonrenew

        if self.flag_nonvanish:
            par = [self.M, self.S, self.K, self.energies, self.tau_inv,
                   self.costs, self.growth, self.C]
            self.dynamics = self.get_vector_field_crossfeeding_off_nonvanish

        Y = odeint(self.dynamics, Y_ini, T, args=(par,), mxstep=5000, atol=1e-6)
        Y[Y < 1e-6] = 0

        self.R_f = Y[-1, 0:self.M]
        self.N_f = Y[-1, self.M:self.M + self.S]
        self.costs_power = self.N_f.dot(self.costs)
        self.eff = self.costs_power / self.power_max
        self.survive = np.count_nonzero(self.N_f)
        self.Lyapunov = self.K.dot(np.log(1. + self.N_f.dot(self.C))) - self.N_f.dot(self.costs)
        return Y[:, 0:self.M], Y[:, self.M:self.M + self.S]

    ##########################################################################
    # ODE right-hand sides
    ##########################################################################

    def get_vector_field_crossfeeding_off(self, Y, t, par):
        """Renewable resource dynamics (linear, Type-I).

        dR/dt = (K - R) * tau_inv - R * (N @ C)
        dN/dt = N * growth * (C @ (energies * R) - costs)
        """
        [M, S, K, energies, tau_inv, Costs, growth, C] = par
        R = Y[0:M]
        R[R < 0] = 0
        N = Y[M:M + S]
        N[N < 0] = 0
        species_vector = N * growth * (C.dot(energies * R) - Costs)
        resource_vector = (K - R) * tau_inv - R * N.dot(C)
        return np.concatenate((resource_vector, species_vector))

    def get_vector_field_crossfeeding_off_nonvanish(self, Y, t, par):
        """Constant resource input — carrying capacity K is not depleted.

        dR/dt = K * tau_inv - R * (N @ C)
        """
        [M, S, K, energies, tau_inv, Costs, growth, C] = par
        R = Y[0:M]
        R[R < 0] = 0
        N = Y[M:M + S]
        N[N < 0] = 0
        species_vector = N * growth * (C.dot(energies * R) - Costs)
        resource_vector = K * tau_inv - R * N.dot(C)
        return np.concatenate((resource_vector, species_vector))

    def get_vector_field_crossfeeding_off_nonrenew(self, Y, t, par):
        """Non-renewable (logistic) resource dynamics.

        dR/dt = R * (K - R) * tau_inv - R * (N @ C)
        """
        [M, S, K, energies, tau_inv, Costs, growth, C] = par
        R = Y[0:M]
        R[R < 0] = 0
        N = Y[M:M + S]
        N[N < 0] = 0
        species_vector = N * growth * (C.dot(energies * R) - Costs)
        resource_vector = R * (K - R) * tau_inv - R * N.dot(C)
        return np.concatenate((resource_vector, species_vector))

    def get_vector_field_crossfeeding_off_nonlinear(self, Y, t, par):
        """Renewable resource dynamics with Type-II (Michaelis-Menten) functional response.

        Uses ``self.gamma`` (Hill coefficient) and ``self.K_sat`` (half-saturation).
        """
        [M, S, K, energies, tau_inv, Costs, growth, C] = par
        R = Y[0:M]
        N = Y[M:M + S]
        Rm = np.power(R, self.gamma)
        Rm_divide = np.divide(Rm, np.add(Rm, self.K_sat))
        species_vector = N * growth * (C.dot(energies * Rm_divide) - Costs)
        resource_vector = (K - R) * tau_inv - Rm_divide * N.dot(C)
        return np.concatenate((resource_vector, species_vector))

    ##############################################################################

    def dynamics_nonrenewable_typeI_crossfeeding_on(self, Y, t, par):
        """Crossfeeding dynamics: species excrete metabolic byproducts recycled via D.

        dR/dt = (R0 - R) * tau_inv - N @ (C * R) + (1 - e) * D @ (N @ (C * R))
        dN/dt = N * growth * (e * C @ R - costs)
        """
        [M, S, R0, energies, tau_inv, costs, growth, C, D, e] = par
        R = Y[0:M]
        N = Y[M:M + S]
        p0 = C * R
        p1 = N.dot(p0)
        resource_production = (1 - e) * D.dot(p1)
        species = N * growth * (e * C.dot(R) - costs)
        resources = (R0 - R) * tau_inv - p1 + resource_production
        return np.concatenate((resources, species))


##########################################################################
# Standalone utility functions
##########################################################################

def Consum_matrix_MA(p, S, M):
    """Generate a random binary S×M consumption matrix with entry probability p."""
    c = np.zeros((S, M))
    for i in range(S):
        for j in range(M):
            if np.random.rand() < p:
                c[i, j] = 1.0
    return c


def Entropy_cal(N):
    """Shannon entropy of a species-abundance vector N."""
    freq = N / np.sum(N)
    return -np.dot(freq, np.log(freq))


##########################################################################
# metabolic_flag = "random" | "fixed" | "tiled" | "one-step"
##########################################################################

def Make_consumption_matrices(S, M, nu, p, q, metabolic_flag):
    """Build a list of S trophic D-matrices using a probabilistic trophic structure.

    Parameters
    ----------
    S : int
        Number of species.
    M : int
        Number of resources.
    nu : int
        Highest trophic layer index for any species.
    p : float
        Probability of leakage at each trophic step.
    q : float
        Probability of adding a metabolic pathway.
    metabolic_flag : str
        One of "random", "fixed", "tiled", or "one-step".

    Returns
    -------
    list of ndarray, each of shape (M, M)
    """
    nu = np.minimum(M - 1, nu)
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
        for a in range(M):
            for b in range(M):
                if a <= b or b < nu_j:
                    D_matrix[a, b] = 0
                else:
                    # Kronecker delta: 1 if a == M-1, else 0
                    kd = 1 if a == M - 1 else 0
                    D_matrix[a, b] = (np.random.binomial(1, q)
                                      * (1 - p) ** (a - b - 1)
                                      * p ** (1 - kd))
        D_matrix[D_matrix < 1e-3] = 0
        ecosystem.append(D_matrix)

    if metabolic_flag == "one-step":
        ecosystem1 = []
        for j in range(M - 1):
            D = np.zeros((M, M))
            D[j + 1, j] = 1
            ecosystem1.append(D)
        return ecosystem1

    return ecosystem
