# -*- coding: utf-8 -*-
"""
Created on Thu 03/31/2019

@author: Wenping Cui
"""
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
from Eco_function.usertools import MakeMatrices
import pdb
import os.path
import pickle
from scipy.integrate import odeint
from multiprocessing import Pool
import multiprocessing 
import argparse
parser = argparse.ArgumentParser(description='Process types and dynamics')
parser.add_argument('--B', default='identity')
parser.add_argument('--C', default='gaussian')
parser.add_argument('--d', default='quadratic')
parser.add_argument('--s', default='CVXOPT')
args = parser.parse_args()
dynamics=  args.d #'quadratic', 'linear','crossfeeding'
B_type = args.B   #'identity', 'null', 'circulant' and 'block'
C_type = args.C   #'gaussian',‘binomial'
Simulation_type=args.s # 'ODE', 'CVXOPT'
save_pkl=False
start_time = time.time()
Pool_num=10
file_name='Community_'+C_type+'_'+B_type +'_'+dynamics+'_'+Simulation_type+'_linear.csv'

parameters = {}
parameters['sample_size']=1000;
parameters['S'] =100;
parameters['M']=100;

parameters['K']=1.0;
parameters['sigma_K']=0.1;

parameters['mu']=1.0;
parameters['sigma_c']=2.0; 

parameters['m']=0.1;
parameters['sigma_m']=0.01;
parameters['loop_size']=50;


parameters['t0']=0;
parameters['t1']=500;
parameters['Nt']=1000;
filename='crossfeeding_D.pkl'
with open(filename, 'rb') as f:
    D = pickle.load(f)
def func_parallel(para):
	parameter = {}
	parameter['sample_size']=para[0];
	parameter['S'] =para[1];
	parameter['M']=para[2];

	parameter['K']=para[3];
	parameter['sigma_K']=para[4];

	parameter['mu']=para[5];
	parameter['sigma_c']=para[6]; 

	parameter['m']=para[7];
	parameter['sigma_m']=para[8];
	parameter['loop_size']=para[9];


	parameter['t0']=para[10];
	parameter['t1']=para[11];
	parameter['Nt']=para[12];

	epsilon=para[13]
	mu=para[14]
	D=para[15]
	parameter['tau_inv']=para[16]
	Model=CRM_Cavity_simulation(parameter)
	Model.Bnormal=False
	Model.gamma_flag='S/M'
	if B_type=='identity': #'diag', 'null', 'circulant' and 'block'
		Model.B_type='identity'
		Model.mu=mu
		Model.epsilon=epsilon
	elif B_type=='null':
		Model.B_type='null'
		Model.mu=mu
		Model.sigma_c=epsilon
	elif B_type=='circulant':
		Model.B_type='circulant'
		Model.mu=mu
		Model.epsilon=epsilon
	elif B_type=='block':
		Model.B_type='block'
		Model.mu=mu
		Model.epsilon=epsilon
	if C_type=='binomial':
		Model.C_type='binomial'
		Model.p_c=epsilon
		Model.mu = para[14]
		Model.epsilon = para[13]
	elif C_type=='gamma':
		Model.C_type='gamma'
		Model.mu=mu
		Model.epsilon=epsilon
	elif C_type=='gaussian':
		Model.C_type='gaussian'
		Model.mu=mu
		Model.epsilon=epsilon
	elif C_type=='uniform':
		Model.C_type='uniform'
		Model.mu=mu
		Model.epsilon=epsilon
	if dynamics=='linear': #'quadratic' 'linear'
		mean_var=Model.ode_simulation(Dynamics=dynamics,Simulation_type=Simulation_type)
	elif dynamics=='quadratic': #'quadratic' 'linear'
		mean_var=Model.ode_simulation(Dynamics=dynamics,Simulation_type=Simulation_type)
	elif dynamics=='crossfeeding': #'quadratic' 'linear'
		Model.D =D
		Model.e=0.6
		Model.flag_crossfeeding=True
		mean_var=Model.ode_simulation(Dynamics=dynamics,Simulation_type=Simulation_type)
	if save_pkl:
		filename='RN_org_'+'_'+dynamics+'_'+C_type +'_'+B_type+'_sigc_'+str(round(epsilon,2))+'_mu_'+str(round(mu,2)) +'_k_'+str(round(parameter['K'],2))+'.pkl'
		with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
			pickle.dump((Model.R_org, Model.N_org, Model.lams), f)
		filename='Clam_org_'+'_'+dynamics+'_'+C_type +'_'+B_type+'_sigc_'+str(round(epsilon,2))+'_mu_'+str(round(mu,2)) +'_k_'+str(round(parameter['K'],2))+'.pkl'
		with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
			pickle.dump((Model.col_lams, Model.col_C, Model.col_C_del), f)
		filename='CRN_org_'+'_'+dynamics+'_'+C_type +'_'+B_type+'_sigc_'+str(round(epsilon,2))+'_mu_'+str(round(mu,2)) +'_k_'+str(round(parameter['K'],2))+'.pkl'
		with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
			pickle.dump((Model.col_R, Model.col_N), f)
	mean_var['dynamics']=dynamics
	mean_var['size']=parameter['S']
	mean_var['mu']=mu
	mean_var['epsilon']=epsilon
	mean_var['sample_size']=parameter['sample_size']
	mean_var['K']=para[3]
	mean_var['M']=para[2];
	mean_var['S']=para[1]
	index = [0]
	para_df = pd.DataFrame(mean_var, index=index)
	print(para_df)
	return para_df

jobs=[];
for S in [100]:
	parameters['S'] =S;
	parameters['M'] =S
	parameters['sample_size']=int(500);
	#for mu in np.append(0,np.logspace(-3.0, 2., num=10)):
	for mu in [1.]:
		parameters['tau_inv']=1.
		for k in [1.]:
			parameters['K']=k
			parameters['mu']=mu
			if C_type == 'gaussian':
				ranges=np.logspace(-6.0,3., num=150)
			elif C_type == 'binomial':
				ranges=np.logspace(-6.0,-0.3, num=160)
				ranges=np.linspace(0.0, 0.5, num=100)
			elif C_type == 'uniform':
				ranges=np.logspace(-6.0, 2, num=160)
				ranges=np.linspace(0.0, 0.5, num=100)
			#ranges=np.linspace(0.0, 2.7, num=28)
			#ranges=[0.1, 0.3, 0.5, 0.7, 1.0, 2., 5., 10., 20., 30.,40.,50.,100.]
			#sranges=np.linspace(0.0, 1.1, num=15)
			for epsilon in ranges:
				jobs.append([parameters['sample_size'],parameters['S'],parameters['M'],parameters['K'],parameters['sigma_K'], parameters['mu'], \
				parameters['sigma_c'],parameters['m'],parameters['sigma_m'],parameters['loop_size'],parameters['t0'],parameters['t1'],parameters['Nt']  \
				,epsilon, mu, D,parameters['tau_inv']])
pool = Pool(processes=Pool_num)
results = pool.map(func_parallel, jobs)
pool.close()
pool.join()
results_df = pd.concat(results)
with open(file_name, 'a') as f:
		results_df.to_csv(f, index=False,encoding='utf-8')




