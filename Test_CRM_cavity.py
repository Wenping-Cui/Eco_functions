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
from Eco_function.Model_cavity_CRM import *
import pdb
import os.path
import pickle
from scipy.integrate import odeint
from multiprocessing import Pool
import multiprocessing 
start_time = time.time()
dynamics='quadratic' #'quadratic' 'linear'
#file_name='Data/sigc_Li.csv'
columns=['mean_N', 'q_N','mean_R','q_R', 'survive','mu','sigc','m','sigm','K','sigK','S','M','gamma']
para_df = pd.DataFrame(columns=columns)
parameters = {}

parameters['sample_size']=1;

parameters['S'] =100;
parameters['M']=100;

parameters['K']=1.;
parameters['sigma_K']=0.1;

parameters['mu']=0.;
parameters['sigma_c']=1.; 

parameters['m']=0.1;
parameters['sigma_m']=0.01;
parameters['loop_size']=50;


parameters['t0']=0;
parameters['t1']=10000;
parameters['Nt']=20000;


print (parameters)


Model=CRM_Cavity_simulation(parameters)
Model.gamma_flag='S/M'
Model.B_type='null'
mean_var=Model.ode_simulation(Dynamics=dynamics, Simulation_type='CVXOPT')
print("cost",Model.costs)
print (mean_var)
print ('consumed power=', mean_var['power']/parameters['S'])


print ('finished time=  ', time.time()-start_time)