# Eco_functions
Simulations for Consumer-Resource Models
# How to install the package
Type in the terminal.

```
git clone https://github.com/Wenping-Cui/Eco_functions
cd Eco_functions
pip install -e .
``` 

# An example to do simulations
``` 
### Import the package in the code:
from Eco_function.eco_lib import *
from Eco_function.eco_plot import *
from Eco_function.eco_func import *
import time
import pandas as pd
import matplotlib
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os.path
import pickle
import seaborn as sns
 
## Initial the number of species and resources
S=500;
M=100;
flag='linear'#'constant', 'linear', 'quadratic' for different dynamics
flag_crossfeeding = False; # simulation with crossfeeding or not.
K =np.random.normal(5, 1, M)  # Initial the number of species and resources
C=np.random.normal(5, 2,[S,M]) # Initial the consumer matrix
energies = np.ones(M)
tau_inv = np.ones(M)
#Ode solver parameter
t0 = 0;
t1 = 1000;
N_t = 10000;
T_par = [t0, t1, N_t];
T = np.linspace(t0, t1, num=N_t);
R_ini = 0.2 * np.ones(M);
N_ini =  0.2 * np.ones(S);
# Start to simulate
costs=np.ones(S)  # Initial the cost
growth=np.ones(S) # Initial the growth rate
# Start to simulate
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
sim_par = [flag_crossfeeding, M, S, R_ini, N_ini,T_par, C, energies, tau_inv, costs, growth, K] 
Model =Ecology_simulation(sim_par)
Model.flag_renew=flag_renew;
Model.flag_nonvanish=flag_nonvanish;
Rt, Nt=Model.simulation()
``` 