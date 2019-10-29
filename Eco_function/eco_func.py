import numpy as np
from scipy.integrate import odeint
import pdb
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors
import math
import pandas as pd
import pickle
import os
def appendDFToCSV_void(df, csvFilePath, sep=","):
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=True, sep=sep)
    else:
        df.to_csv(csvFilePath, mode='a', index=True, sep=sep, header=False)

def load_parameters(filename):
	#sim_par=[K, tau_inv, energies,c_pool, costs_pool, growth_pool]
	if os.path.exists(filename):
		sim_par=pickle.load( open( filename, "rb" ) )
		return 'File exists and load existed parameter',sim_par   	
	else:
		return 'File does not exist', None
def save_parameters(sim,filename):
 	with open(filename, 'wb') as f:
    		pickle.dump(sim, f)       
    		return 'File is saved' 	
def KL_lams(lams, sigc, phiN,phiR, M, Nbins=100):
	s=sigc*np.sqrt(M)
	c=phiN/phiR
	a=(s**2)*(1-np.sqrt(c))**2
	b=(s**2)*(1+np.sqrt(c))**2
	bins=np.linspace(a,b,Nbins)
	p,bins =np.histogram(lams, bins=bins, density=True)
	bins=np.linspace(a,b,Nbins-1)
	q=MP_density(bins, a, b, c, s)
	q=np.nan_to_num(q)
	return KL(p,q)


def MP_density(bins, a,b,c, s):
	return (1./(2*np.pi*bins*c*s**2))*np.sqrt((b-bins)*(bins-a))

def KL(P,Q,epsilon=1e-5):
# Epsilon is used here to avoid conditional code for
# checking that neither P nor Q is equal to 0. 
    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon
    return np.sum(P*np.log(P/Q))