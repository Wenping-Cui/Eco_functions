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