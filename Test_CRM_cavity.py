import time
import numpy as np
from Eco_function.Model_cavity_CRM import CRM_Cavity_simulation
start_time = time.time()
dynamics='quadratic'  # 'quadratic', 'linear', or 'crossfeeding'
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