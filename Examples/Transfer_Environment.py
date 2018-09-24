# Transfer species into different environment
import pandas as pd
import numpy as np 
import pickle
import re
from tqdm import tqdm
from Eco_function.eco_lib import *
from Eco_function.eco_plot import *
from Eco_function.eco_func import *

local_Initial_type='steady_initial'
local_Resource_type='quadratic' #'quadratic'
New_Initial_type='RN_constant_initial'
New_Resource_type= 'linear' #'linear'
final_step=299
save_file='simulation_data/Transfer_environment.csv'  
columns=['Local_initial_type', 'Local_Resource_Type','Local_richness', 'New_initial_type', 'New_Resource_Type','New_richness','consumed power']
data_save = pd.DataFrame(columns=columns)
# Read the data files about speices and resources pool
file_name='simulation_data/Richness_augmentation2018_04_11-18-15.pkl'
message,sim_par=load_parameters(file_name)
[N_t, M, p, flag_crossfeeding, costs_pool, growth_pool, c_pool, K, energies, tau_inv, t0, t1, Nt, T_par]=sim_par

# Read the data files about the invasion simulation
data_Community_augmentation='simulation_data/Richness_augmentation_p0.1_2018_04_16-13-52_correction.csv'
df = pd.read_csv(data_Community_augmentation)
data=df.loc[(df['step']==final_step) & (df['Initial_type']==local_Initial_type) &(df['Resource Type']==local_Resource_type)]

flag=New_Resource_type
initial_type=New_Initial_type

length=(len(data.index))
for i in tqdm(range(length)):   
    index=data.index[i]
    array_Survie=list(data[data.index==index]['Survie specie order'])[0]
    array_Survie1=np.asfarray(re.findall(r"[-+]?\d*\.\d+|\d+", array_Survie))
    order=array_Survie1.astype(int)
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
    R,N=0,0;
    for i in order:
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
    sim_par = [flag_crossfeeding, M, S, R_ini, N_ini,T_par, C, energies, tau_inv, costs, growth, K] 
    Model =Ecology_simulation(sim_par)
    Model.flag_renew=flag_renew;
    Model.flag_nonvanish=flag_nonvanish;
    R_t, N_t=Model.simulation()  
    R, N=Model.R_f, Model.N_f; 
    data_current=pd.DataFrame([[local_Initial_type, local_Resource_type, S, New_Initial_type,New_Resource_type,  Model.survive, Model.costs_power]],columns=columns)
    data_save = pd.concat([data_save , data_current], ignore_index=True)
        
data_save.to_csv(save_file, index=False, encoding='utf-8')   