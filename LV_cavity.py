# -*- coding: utf-8 -*-
"""
Created on Thu 03/31/2019

@author: Wenping Cui
"""
import time
import pandas as pd
import pdb
import matplotlib
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import itertools
from Eco_function.eco_lib import *
from Eco_function.Model_cavity_LV import *
import pdb
import os.path
import pickle
from scipy.integrate import odeint
from multiprocessing import Pool
import multiprocessing 
import argparse
#pdb.set_trace()

parser = argparse.ArgumentParser(description='Process types and dynamics')
parser.add_argument('--A', default='gaussian')
parser.add_argument('--B', default='identity')



args = parser.parse_args()
A_type = args.A   #'gaussian',‘binomial'
B_type = args.B   #'identity', 'null', 'circulant' and 'block'


start_time = time.time()
Pool_num=10
file_name='LV_'+A_type+'_sig_1.csv'

parameters = {}
parameters['sample_size']=10
parameters['S'] =100 
parameters['A_type']=A_type
parameters['B_type']=B_type

parameters['k']=1.0;
parameters['sigma_k']=0.1;

parameters['mu']=1.0;
parameters['epsilon'] =0.1

parameters['g']=1.;
parameters['sigma_g']=0.;

parameters['B']=0

parameters['ODE_Time']=[0,500, 2000]

parameters['mapping_CR']=False

parameters['sys_gamma']=0


def func_parallel(para):
	index=para[0]
	paras={}
	paras={**para[1],**paras}
	start_time0=time.time()
	S=paras['S']
	assert paras['A_type'] in ['binomial','gamma', 'gaussian','uniform'], \
	"A type must be 'binomial','gamma' ,'gaussian' or 'uniform'"

	assert paras['B_type'] in ['identity','null', 'circulant','block'], \
	"B type must be 'identity','null', 'circulant','block'"

	if paras['B_type']=='identity': #'diag', 'null', 'circulant' and 'block'
		B=np.identity(S)
	elif paras['B_type']=='null':
		B=0
	elif paras['B_type']=='circulant':
		D = [7, 1]  # generalist, specialist
		B=circ(S, D[1])+np.identity(S)
	elif paras['B_type']=='block':
		B=block(int(S/10), 10)+np.identity(S)

	paras['B']=B

	Model=LV_Cavity_simulation(paras)
	mean_var=Model._simulation(dynamics="ODE")
	epsilon, mu, gamma = paras['epsilon'], paras['mu'], paras['sys_gamma']
	save_pkl =1
	if save_pkl:
		filename='eigenvalues'+'_'+A_type +'_'+B_type+'_sigc_'+str(round(epsilon,2))+'_mu_'+str(round(mu,2))+"gamma_"+str(round(gamma,2))+'.pkl'
		with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
			pickle.dump((Model.lams, Model.lams_org, Model.phin_list, Model.col_N), f)

	paras.pop("B", None)
	paras.pop("ODE_Time", None)
	data= { **paras,**mean_var}
	para_df = pd.DataFrame(data, index=[index])
	print("index", index)
	print("*"*20)
	print("finished time: ", time.time()-start_time0)
	return para_df


jobs=[];
index=0
for B_type in ['identity']:
	for S in [100]:
		for epsilon in np.arange(0.4, 1., 0.1): 
			for mu in [0]:
				#np.arange(-1, 1.1, 0.1)
				for sys_gamma in [1.,0.]:
					parameters['S'] =S
					parameters['sys_gamma'] = sys_gamma
					parameters['sample_size']=int(1000)
					parameters['B_type']=B_type
					parameters['mu']=mu
					parameters['epsilon'] =epsilon  
					parameters['sys_gamma']= sys_gamma
					var=parameters.copy()
					jobs.append([index, var])
					index=index+1
pool = Pool(processes=Pool_num)
results = pool.map(func_parallel, jobs)
pool.close()
pool.join()
results_df = pd.concat(results)
with open(file_name, 'a', newline = '\n') as f:
		results_df.to_csv(f, index=False,encoding='utf-8')

print ('finish time',int(time.time() - start_time))



