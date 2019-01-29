import numpy as np
from scipy.integrate import odeint
import pdb
import time
#pdb.set_trace()
##########################################################################


class Ecology_simulation(object):
    def __init__(self, par):
        self.flag_crossfeeding=par[0]
        self.M = par[1];
        self.S = par[2];
        self.R_ini = par[3]; 
        self.N_ini = par[4];
        self.t0, self.t1, self.Nt = par[5];
        self.C = par[6]; 
        self.energies = par[7]; 
        self.tau_inv = par[8]; 
        self.costs= par[9];
        self.growth = par[10];
        if self.flag_crossfeeding:
            self.D = par[11];
            self.non_zero_resource =par[12];
            self.resource_amount = par[13];
            self.K=self.resource_amount
            self.power_max = np.dot(np.dot(self.resource_amount,self.energies[self.non_zero_resource]),self.tau_inv[self.non_zero_resource]);
        else:
            self.K= par[11];
            self.power_max =np.dot(self.K*self.tau_inv, self.energies);
        self.costs_power=0;
        self.eff=0;
        self.R_f = np.zeros(self.M);
        self.N_f = np.zeros(self.S);
        self.survive =0;
        self.flag_renew=False
        self.flag_linear=False;
        self.gamma=1;
        self.K_sat=1;
        self.flag_nonvanish=False;

    def simulation(self,):
        Y_ini = np.concatenate((self.R_ini, self.N_ini))
        T = np.linspace(self.t0, self.t1, num=self.Nt);
        self.costs= np.asarray(self.costs);
        self.C = np.asarray(self.C);


        if self.flag_crossfeeding:
            self.DcE = np.zeros((self.S,self.M));
            self.D = np.asarray(self.D);
            for i in range(self.S):
                for alpha in range(self.M):
                         self.DcE[i,alpha]= self.C[i, alpha]*(self.energies[alpha]- np.sum(self.D[alpha, :]*self.energies[:]))
            self.R0 = np.zeros(self.M);
            self.R0[self.non_zero_resource] = self.resource_amount;         
            self.dynamics = self.dynamics_nonrenewable_typeI_crossfeeding_on
            par = [self.M, self.S, self.R0, self.energies, self.tau_inv, self.costs, self.growth, self.C, self.D, self.DcE] 
        else: 
            if self.flag_renew:
                if self.flag_linear:
                    par = [self.M, self.S, self.K, self.energies, self.tau_inv, self.costs, self.growth, self.C]
                    self.dynamics = self.get_vector_field_crossfeeding_off
                else:    
                    par = [self.M, self.S, self.K, self.energies, self.tau_inv, self.costs, self.growth, self.C]
                    self.dynamics = self.get_vector_field_crossfeeding_off_nonlinear
            else:    
                par = [self.M, self.S, self.K, self.energies, self.tau_inv, self.costs, self.growth, self.C]
                self.dynamics = self.get_vector_field_crossfeeding_off_nonrenew 
        if self.flag_nonvanish:
            par = [self.M, self.S, self.K, self.energies, self.tau_inv, self.costs, self.growth, self.C]
            self.dynamics = self.get_vector_field_crossfeeding_off_nonvanish


        Y = odeint(self.dynamics, Y_ini, T, args=(par,),mxstep=5000, atol=10 ** -6)
        Y[np.where(Y < 10 ** -6)] = 0


        self.R_f = Y[-1, 0:self.M]
        self.N_f = Y[-1, self.M:self.M + self.S]
        self.costs_power = self.N_f.dot(self.costs)
        self.eff = self.costs_power/self.power_max;
        self.survive = np.count_nonzero(self.N_f)   
        self.Lyapunov =self.K.dot(np.log(1.+self.N_f.dot(self.C)))-self.N_f.dot(self.costs); 
        return Y[:, 0:self.M], Y[:, self.M:self.M + self.S]


    def get_vector_field_crossfeeding_off(self, Y, t, par):
        [M, S, K, energies, tau_inv, Costs, growth, C] = par;
        R = Y[0:M]
        R[np.where(R < 0)] = 0;
        N = Y[M:M + S]
        N[np.where(N < 0)] = 0;
        species_vector = N * growth * ((C.dot(energies * R)) - Costs) 
        resource_vector = (K - R) * tau_inv - R * N.dot(C) 
        output_vector = np.concatenate((resource_vector, species_vector));
        return output_vector
    def get_vector_field_crossfeeding_off_nonvanish(self, Y, t, par):
        [M, S, K, energies, tau_inv, Costs, growth, C] = par;
        R = Y[0:M]
        R[np.where(R < 0)] = 0;
        N = Y[M:M + S]
        N[np.where(N < 0)] = 0;
        species_vector = N * growth * ((C.dot(energies * R)) - Costs) 
        resource_vector = (K) * tau_inv - R * N.dot(C) 
        output_vector = np.concatenate((resource_vector, species_vector));
        return output_vector    
    def get_vector_field_crossfeeding_off_nonrenew(self, Y, t, par):
        [M, S, K, energies, tau_inv, Costs, growth, C] = par;
        R = Y[0:M]
        R[np.where(R < 0)] = 0;
        N = Y[M:M + S]
        N[np.where(N < 0)] = 0;
        species_vector = N * growth * ((C.dot(energies * R)) - Costs) 
        resource_vector = R*(K - R) * tau_inv - R * N.dot(C) 
        output_vector = np.concatenate((resource_vector, species_vector));
        return output_vector  
    def get_vector_field_crossfeeding_off_nonlinear(self, Y, t, par):
        [M, S, K, energies, tau_inv, Costs, growth, C] = par;
        R = Y[0:M]
        #R[np.where(R < 10 ** -3)] = 0;
        N = Y[M:M + S]
        Rm = np.power(R, self.gamma)
        Rm_divide = np.divide(Rm, np.add(Rm, self.K_sat));
        #N[np.where(N < 10 ** -3)] = 0;
        species_vector = N * growth * ((C.dot(energies * Rm_divide )) - Costs) 
        resource_vector = (K - R) * tau_inv - Rm_divide * N.dot(C) 
        output_vector = np.concatenate((resource_vector, species_vector));
        return output_vector     


   ##############################################################################  
    def dynamics_nonrenewable_typeI_crossfeeding_on(self, Y, t, par):
        [M, S, R0, energies, tau_inv, costs, growth, C, D, DcE] = par
        R = Y[0:M]
        N = Y[M:M + S]
        p0 = C * R
        p1 = N.dot(p0)
        resource_production = D.dot(p1)
        species = N*growth*((DcE.dot(R))-costs)
        resources =R*(R0-R)*tau_inv - p1 + resource_production
        output = np.concatenate((resources, species));
        return output

    def test(self,):
        Y_ini = np.concatenate((self.R_ini, self.N_ini, self.Q_ini))
   


##########################################################################
def K_levy(M):
    K = np.zeros(M);
    for i in range(len(K)):
        k = levy.rvs(loc=0, scale=1, size=1, random_state=None);
        if k<0.1:
           k = 0;
        K[i] = k;
    return K

##########################################################################
def Consum_matrix_MA(p, S, M):
    c = np.zeros((S, M));
    for i in range(S):
        for j in range(M):
            if np.random.rand() < p:
                  c[i,j]= 1.0;
    return c

def Entropy_cal(N):
    return -np.dot(N/np.sum(N),np.log(N/np.sum(N)))
##########################################################################
# metabolic_flag="random", "fixed", "tiled"
def Make_consumption_matrices(S, M, nu, p, q, metabolic_flag):
    """M- number of resources; nu- highest trophic layer for any species; p-probability of leakage; q-prob of adding pathway; There is always
    atleast one species at trophic level nu rest of nu (top trophic layer) are randomly drawn between nu and M;
    """
    # Create master D matrix for vu=0 and q=1 for fixed p
    delta = lambda x, y: 1 if x == y else 0;
    nu = np.minimum(M - 1, nu);
    if metabolic_flag == "random":
        nu_array = np.append(np.array([nu]),np.random.randint(nu,M-1,size=S-1))

    if metabolic_flag == "fixed":
        nu_array = nu * np.ones(S)

    if metabolic_flag == "tiled":
        nu_array = np.mod(np.arange(nu, S + 1) - nu, M - nu) + nu

    if metabolic_flag == "one-step":
        nu_array = nu * np.ones(S)
    ecosystem = []
    for j in range(0, S):
        nu = nu_array[j]
        D_matrix = np.zeros((M, M));
        for a in range(0, M):
            for b in range(0, M):
                if a <= b or b < nu:
                    D_matrix[a, b] = 0
                else:
                    D_matrix[a, b] = np.random.binomial(1, q) * (1 - p) ** (a - b - 1) * p ** (1 - delta(a, M - 1))
        D_matrix[np.where(D_matrix < 10 ** -3)] = 0;
        ecosystem.append(D_matrix)
    if metabolic_flag == "one-step":
        ecosystem1 = [];
        for j in range(M-1):
            D = np.zeros((M, M))
            D[j + 1, j] = 1
            ecosystem1.append(D)
    return ecosystem1 if metabolic_flag == "one-step" else ecosystem

def combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)



  
