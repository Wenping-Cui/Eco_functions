import numpy as np
from past.builtins import xrange
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



##########################################################################
def Consum_matrix_MA(p, S, M):
    c = np.zeros((S, M));
    for i in range(S):
        for j in range(M):
            if np.random.rand() < p:
                  c[i,j]= 1.0;
    return c