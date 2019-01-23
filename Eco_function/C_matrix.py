import numpy as np
from past.builtins import xrange
from scipy.linalg import block_diag
import math as mt
def Consum_matrix(c_m, c_sigma, q_c, M, F, flag_family):
    c = np.zeros(M);
    for j in range(M):
            if j%F==flag_family:
               c[j] =np.abs(np.random.normal(c_m, c_sigma)*q_c)
            else:  
               c[j] =np.abs(np.random.normal(c_m, c_sigma)*(1-q_c))
    c = c/np.sum(c)*M*c_m       
    c[0] = np.abs(np.random.normal(c_m, c_sigma))           
    return c

def block(n, m): # m: block size n: number of blocks
    a = np.full((m, m), 1)
    C = block_diag(*([a] * n))
    return C

def circ(n, r):
    C_type = 'gaussian2'
    cc = np.array([])
    for i in np.arange(n):

        if C_type == 'gaussian2':
            cc = np.append(cc, (mt.exp(-min(i, abs(n-i))**2 / (2 * r**2))))

    C = cc
    for i in np.arange(n-1):
        cc = np.append(cc[-1:], cc[:-1])
        C  = np.vstack((C, cc))
    return C

def binomial(p, S, M):
    c = np.zeros((S, M));
    for i in range(S):
        for j in range(M):
            if np.random.rand() < p:
                  c[i,j]= 1.0;
    return c