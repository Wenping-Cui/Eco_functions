import time
import pandas as pd
import matplotlib
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import itertools
from Eco_function.eco_lib import *
from Eco_function.eco_plot import *
from Eco_function.eco_func import *
import pdb
import os.path
import pickle
from scipy.integrate import odeint
from multiprocessing import Pool
import multiprocessing 
start_time = time.time()
load_parameter=True;
initial_type_list=['R_constant_initial', 'N_constant_initial', 'RN_constant_initial','steady_initial', 'large_initial']
Pool_num=28;
NN=100; # number of repeated invasion experiments
tableau20=color20() # initial colors for plot;
columns=['Initial_type', 'Resource Type','step','richness', 'consumed power','Community augmentation','Rejection failure','Indirect failure', 'Replacement', 'Extinction','Survie specie order','Survie species abuncance', 'Invasive Species NO']
######################
# Global Parameters
######################
N_T = 2000  # total number of species in the pool
N_s =300; # choose the size of speices from the pool
M = 100;  # number of resources
p = 0.1; # probability for generationg nonzero elements in consumer matrix
flag_crossfeeding = False; # simulation with crossfeeding or not.
#################################
# Build Species Pool
##################################
costs_pool=np.ndarray.tolist(1.0+np.random.rand(N_T)*2)
growth_pool=np.ndarray.tolist(np.ones(N_T))
c_pool = Consum_matrix_MA(p, N_T, M)
#Ode solver parameter
t0 = 0;
t1 = 2000;
Nt = 4000;
T_par = [t0, t1, Nt];

#################################
# RESOURCE PROPERTIES
##################################
K =np.abs(10*np.random.randn(M)+1);
#K =np.abs(5*np.ones(M)+1);
#Creat energy vector
deltaE = 1.0;
energies = deltaE*np.random.rand(M)+1.0
tau_inv = 1 * np.ones(M)
timestr = time.strftime("%Y_%m_%d-%H-%M")
if load_parameter:
    file_name='simulation_data/Richness_augmentation2018_04_11-18-15.pkl'
    message,sim_par=load_parameters(file_name)
    [N_T, M, p, flag_crossfeeding, costs_pool, growth_pool, c_pool, K, energies, tau_inv, t0, t1, Nt, T_par]=sim_par
else:     
    sim_par=[N_T,  M, p, flag_crossfeeding, costs_pool, growth_pool, c_pool, K, energies, tau_inv, t0, t1, Nt, T_par]
    filename='simulation_data/Richness_augmentation'+timestr+'.pkl'
    save_parameters(sim_par,filename)
file_name='simulation_data/Richness_augmentation'+'_'+'p'+str(p)+'_'+'2018_04_14-15-47'+'.csv'
# df = pd.DataFrame(columns=columns)
#with open(file_name, 'a') as f:
#        df.to_csv(f, header=True,encoding='utf-8')  
def func_parallel(para):
    global N_T,  M, p, flag_crossfeeding, costs_pool, growth_pool, c_pool, K, energies, tau_inv, t0, t1, Nt, T_par
    initial_type=para[0]
    flag=para[1]
    order=para[2];
    Survive_order=[];
    if flag=='constant':
        flag_nonvanish=True;
        flag_renew=True;
        label='constant'
    elif flag=='linear':
        flag_renew=True;
        flag_nonvanish=False;
        label='linear'
    elif flag=='quadratic':
        flag_renew=False;
        flag_nonvanish=False;
        label='quadratic'  
    TT = [];
    growth =[];
    costs =[];
    C = []; 
    S=0
    ss=0;
    para_df = pd.DataFrame(columns=columns)
    R,N=0,0;
    # simulate invasion processes
    for i in order:
        # randomly choose invasive species from the species pool
        Survive_order.append(i)
        S=S+1                         
        C.append(c_pool[i]);
        costs.append(costs_pool[i])
        growth.append(growth_pool[i])
        # initial the species
        R_ini = 0.1 * np.ones(M);
        N_ini = 0.1 * np.ones(S);
        if initial_type=='RN_constant_initial': # 'constant_initial', 'steady_initial', 'large_initial'
            pass;
        elif initial_type=='R_constant_initial': 
            if S>1 and ss>0: 
                N_ini[:-1]=N;
        elif initial_type=='N_constant_initial':    
            if S>1 and ss>0: 
                R_ini=R;
        elif initial_type=='steady_initial':
            if S>1 and ss>0:
                R_ini = R;
                N_ini[:-1]=N;
        elif initial_type=='large_initial':   
            if S>1 and ss>0:
                R_ini = R;
                N_ini[:-1]=N;  
                N_ini[-1] =100.;         
        # Start to simulate
        sim_par = [flag_crossfeeding, M, S, R_ini, N_ini,T_par, C, energies, tau_inv, costs, growth, K] 
        Model =Ecology_simulation(sim_par)
        Model.flag_renew=flag_renew;
        Model.flag_nonvanish=flag_nonvanish;
        R_t, N_t=Model.simulation()
        
        
        # Output simulation results
        R, N=Model.R_f, Model.N_f; 
        if np.allclose(N_t[-10], N_t[-1], rtol=1e-03, atol=1e-08):
            sim_par = [flag_crossfeeding, M, S, R, N,[t0, 2*t1, Nt], C, energies, tau_inv, costs, growth, K] 
            Model =Ecology_simulation(sim_par)
            Model.flag_renew=flag_renew;
            Model.flag_nonvanish=flag_nonvanish;
            R_t, N_t=Model.simulation()
            R, N=Model.R_f, Model.N_f;
        Augmentation,Rejection_f,Indirect_f, Replacement,Extinction=0, 0, 0, 0,0;
        Compare_R=np.allclose(np.sum(N_ini[:-1]), np.sum(N[:-1]), rtol=1, atol=1e-08);
        if Model.N_f[-1]>0 and Model.survive<S:
            Extinction=S-Model.survive
            Replacement=1;
        elif  Model.N_f[-1]>0 and S==Model.survive:
            Augmentation=1;
        elif  Model.N_f[-1]==0 and Compare_R:
             Rejection_f=1;
        elif  Model.N_f[-1]==0 and (not Compare_R):    
             Indirect_f=1          
        # Delete nonsurviors species
        del_indices = np.where(N == 0)[0]
        for h in sorted(del_indices, reverse=True):
                 del C[h],growth[h],costs[h],Survive_order[h]      
        N = [h for j, h in enumerate(N) if j not in del_indices]
        R_ini = R;
        survive=Model.survive
        S=survive
        para_df_current = pd.DataFrame([[initial_type, flag,ss,S, Model.costs_power, Augmentation,Rejection_f,Indirect_f, Replacement, Extinction, list(Survive_order), N, i]],columns=columns)
        para_df =pd.concat([para_df,para_df_current],ignore_index=True) 
        ss = ss+1  
    return index, para_df 


#####################################################################
ready_list = []
def dummy_func(results):
    global ready_list
    ready_list.append(results[0])  
    df=results[1]
    with open(file_name, 'a') as f:
        df.to_csv(f, index=False, header=False,encoding='utf-8')  

jobs=[];   
index=0;
for n in range(NN):
    for initial_type in np.random.choice(['R_constant_initial', 'N_constant_initial', 'RN_constant_initial','steady_initial', 'large_initial'], 5):
        for R_type in np.random.choice(['linear', 'quadratic', 'constant'], 3):
            jobs.append([initial_type, R_type, np.random.choice(N_T, N_s, replace=False),index])
            index=index+1
pool = Pool(processes=Pool_num,maxtasksperchild=10)
result = {}
print ('total number of jobs',len(jobs))
for index in range(len(jobs)):
    result[index] = (pool.apply_async(func_parallel, (jobs[index],), callback=dummy_func))
    for ready in ready_list:
        result[ready].wait()
        del result[ready]  
    ready_list = []

# clean up
pool.close()
pool.join()