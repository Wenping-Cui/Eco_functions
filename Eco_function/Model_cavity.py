import time
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from Eco_function.eco_lib import *
from Eco_function.eco_func import *
from scipy.integrate import odeint
from scipy.integrate import quad
import random as rand
class Cavity_simulation(object):
	def __init__(self, parameters):
		self.parameters=parameters
		self.S=parameters['S']
		self.M=parameters['M']
		self.K=parameters['K']
		self.sigma_K=parameters['sigma_K']
		self.mu=parameters['mu']
		self.sigma_c=parameters['sigma_c']
		self.cost=parameters['m']
		self.sigma_m=parameters['sigma_m']
		self.sample_size=parameters['sample_size']
		self.loop_size=parameters['loop_size']
		self.Metabolic_Tradeoff=False
		self.binary_c=False
		self.p_c=0.2
		self.epsilon=10**(-3)
	def initialize_random_variable(self,):
		self.flag_crossfeeding = False; # simulation with crossfeeding or not.
		#################################
		# RESOURCE PROPERTIES
		##################################
		self.Ks =np.random.normal(self.K, self.sigma_K, self.M) 

		#Creat energy vector
		self.deltaE = 1.0;
		self.energies = self.deltaE*np.ones(self.M)
		self.tau_inv = np.ones(self.M)
		#################################
		# Build Species Pool
		##################################
		self.growth=np.ones(self.S)
		if self.binary_c:
			self.C = np.random.binomial(1, self.p_c, [self.S,self.M])
		if self.gamma_flag=='S/M':
			self.C=np.random.normal(self.mu/self.M, self.sigma_c/np.sqrt(self.M), [self.S,self.M])
		if self.gamma_flag=='M/S':
			self.C=np.random.normal(self.mu/self.S, self.sigma_c/np.sqrt(self.S), [self.S,self.M])
		if self.Metabolic_Tradeoff:
			self.costs=np.sum(self.C, axis=1)+self.epsilon*np.random.normal(0, 1, self.S)
		else:
			self.costs=np.random.normal(self.cost, self.sigma_m, self.S)		#Ode solver parameter
		self.t0 = 0;
		self.t1 = self.parameters['t1'];
		self.Nt = self.parameters['Nt']
		self.T_par = [self.t0, self.t1, self.Nt];

		self.R_ini=0.1*np.ones(self.M)
		self.N_ini=0.1*np.ones(self.S)
		self.sim_pars = [self.flag_crossfeeding, self.M, self.S, self.R_ini, self.N_ini,self.T_par, self.C, self.energies, self.tau_inv, self.costs, self.growth, self.Ks] 
		return self.sim_pars

	def ode_simulation(self,plot=False, Dynamics='linear', Initial='Auto', Simulation_type='ODE'): 
		phi_R_list=[];
		phi_N_list=[];
		R_list=[];
		N_list=[];
		R_list_bar=[];
		N_list_bar=[];
		qR_list_bar=[];
		qN_list_bar=[];
		phi_R_list_bar=[];
		phi_N_list_bar=[];
		Survive_list=[]
		power=[];
		N_survive_list=[];
		Opti_f=[]
		Growth=[]
		for step in range(self.sample_size):	
			if Initial=='Auto':
				self.sim_pars=self.initialize_random_variable()
			if Initial=='Manually':
				self.sim_pars = [self.flag_crossfeeding, self.M, self.S, self.R_ini, self.N_ini,self.T_par, self.C, self.energies, self.tau_inv, self.costs, self.growth, self.Ks]

			if Simulation_type=='ODE':
				Model =Ecology_simulation(self.sim_pars)
				if Dynamics=='linear':
					Model.flag_nonvanish=False;
					Model.flag_renew=True;
				if Dynamics=='constant':
					Model.flag_nonvanish=True;
				elif Dynamics=='quadratic':
					Model.flag_renew=False;
					Model.flag_nonvanish=False;
				Model.simulation()
				self.R_f, self.N_f=Model.R_f, Model.N_f;
				R, N=Model.R_f, Model.N_f;
				Model_survive=Model.survive;
				Model_costs_power=Model.costs_power
			if Simulation_type=='QP':
				R, N=self.Quadratic_programming(self,)
				R[np.where(R < 10 ** -6)] = 0
				N[np.where(N < 10 ** -6)] = 0
				Model_costs_power=N.dot(self.costs)
				Model_survive=np.count_nonzero(N)
			Opti_f.append((np.linalg.norm(self.Ks-R))**2/self.M)
			Growth.extend(np.dot(self.C,R)-self.costs)
			Survive_list.append(Model_survive)
			phi_R_list.append(np.count_nonzero(R)/float(self.M));
			phi_N_list.append(Model_survive/float(self.S));
			R_list.extend(R)
			N_list.extend(N)
			R_list_bar.append(np.mean(R))
			N_list_bar.append(np.mean(N))
			qR_list_bar.append(np.mean(R**2))
			qN_list_bar.append(np.mean(N**2))
			power.append(Model_costs_power)
			N=N[np.where(N > 0)]
			N_survive_list.extend(N)
		self.mean_R, self.var_R=np.mean(R_list), np.var(R_list)
		self.mean_N, self.var_N=np.mean(N_list), np.var(N_list)
		self.Survive=np.mean(Survive_list)
		self.mean_var_simulation={};
		self.mean_var_simulation['phi_R']=np.mean(phi_R_list)
		self.mean_var_simulation['phi_N']=np.mean(phi_N_list)
		self.mean_var_simulation['mean_R']=self.mean_R
		self.mean_var_simulation['mean_N']=self.mean_N
		self.mean_var_simulation['q_R']=self.var_R+self.mean_R**2
		self.mean_var_simulation['q_N']=self.var_N+self.mean_N**2
		self.mean_var_simulation['Survive']=np.mean(Survive_list)
		self.mean_var_simulation['Survive_bar']=np.std(Survive_list)
		self.mean_var_simulation['phi_R_bar']=np.std(phi_R_list)
		self.mean_var_simulation['phi_N_bar']=np.std(phi_N_list)
		self.mean_var_simulation['mean_R_bar']=np.std(R_list_bar)
		self.mean_var_simulation['mean_N_bar']=np.std(N_list_bar)
		self.mean_var_simulation['q_R_bar']=np.std(qR_list_bar)
		self.mean_var_simulation['q_N_bar']=np.std(qN_list_bar)
		self.mean_var_simulation['var_R']=self.var_R
		self.mean_var_simulation['var_N']=self.var_N
		self.mean_var_simulation['power']=np.mean(power)
		self.mean_var_simulation['power_bar']=np.std(power)
		self.mean_var_simulation['opti_f']=np.mean(Opti_f)
		self.mean_var_simulation['opti_f_bar']=np.std(Opti_f)
		self.N_survive_List=N_survive_list
		self.phir_list=phi_R_list
		self.phin_list=phi_N_list
		self.N_List=N_list
		self.R_List=R_list
		self.G_List=Growth
		if plot:
			num_bins=100
			plt.close('all')
			f, (ax1, ax2, ax3) = plt.subplots(1, 3)
	

			n, bins, patches = ax1.hist(N_survive_list, num_bins, normed=1, facecolor='green', alpha=0.5)
			ax1.set_xlabel('Surviving Species Abundance')
			ax1.set_ylabel('Probability density')
			ax1.set_title(r'Histogram of Species')

			n, bins, patches = ax2.hist(R_list, num_bins, normed=1, facecolor='green', alpha=0.5)
			ax2.set_xlabel('Resources Abundance')
			ax2.set_ylabel('Probability density')
			ax2.set_title(r'Histogram of Resources')


			n, bins, patches = ax3.hist(Growth, num_bins, normed=1, facecolor='green', alpha=0.5)
			ax3.set_xlabel('Growth Rate')
			ax3.set_ylabel('Probability density')
			ax3.set_title(r'Histogram of Growth Rates')
			f.tight_layout()
			return f
		else:
			return self.mean_var_simulation
	def cavity_solution(self,):
		gamma = self.M/self.S;
		var_K=self.sigma_K**2
		var_c=self.sigma_c**2
		var_m=self.sigma_m**2

		mean_N=self.mean_var_simulation['mean_N'];
		mean_R=self.mean_var_simulation['mean_R'];
		var_N = self.mean_var_simulation['var_N'];
		var_R = self.mean_var_simulation['var_R'];
		chi=np.random.randn()
		for l in range(1, self.loop_size):
			var_N_old =var_N 

			q_R = var_R+mean_R**2

			sigma_z = np.sqrt(gamma*var_c*q_R+var_m)

			mean_R=self.K/(1.+gamma*self.mu*mean_N)


			d=(gamma*self.mu*mean_R-self.cost)/sigma_z

			phi_N=self.ifunc(0, d)

			nu= - phi_N/(gamma*var_c*chi)

			chi=self.K/(1.+self.mu*mean_N)**2+3*nu*var_c*self.K**2/(8*(1.+self.mu*mean_N)**4)


			mean_N=sigma_z*self.K/(gamma*var_c*(mean_R**2+var_R))*self.ifunc(1, d)

			var_N=(sigma_z*self.K/(gamma*var_c*(mean_R**2+var_R)))**2*self.ifunc(2, d)-mean_N**2


			#var_R =mean_R**2/self.K**2*var_K+mean_R**4/self.K**2*(self.mu**2*var_N+mean_N**2*var_c)
			var_R =1/(1.+self.mu*mean_N)**2*var_K+nu*var_c/(4*(1.+self.mu*mean_N)**3)*var_K**2


			err = np.abs(var_N - var_N_old)
		print ('error is', err, var_N, var_N_old )
		self.mean_var_cavity={};
		self.mean_var_cavity['mean_R']=mean_R
		self.mean_var_cavity['mean_N']=mean_N
		self.mean_var_cavity['q_R']=var_R+mean_R**2
		self.mean_var_cavity['q_N']=var_R+mean_N**2
		self.mean_var_cavity['var_R']=var_R
		self.mean_var_cavity['var_N']=var_N
		self.mean_var_cavity['Survive']=phi_N*self.Survive
		return self.mean_var_cavity
	def Quadratic_programming(self, Initial='Auto'):
		from cvxopt import matrix
		from cvxopt import solvers
		if Initial=='Auto':
			self.sim_pars=self.initialize_random_variable()
		if Initial=='Manually':
			self.sim_pars = [self.flag_crossfeeding, self.M, self.S, self.R_ini, self.N_ini,self.T_par, self.C, self.energies, self.tau_inv, self.costs, self.growth, self.Ks] 	
		# Define QP parameters (directly)
		M = np.identity(self.M)
		P = np.dot(M.T, M)
		q = -np.dot(self.Ks,M).reshape((self.M,))
		G1= self.C
		h1= self.costs

		G2= -np.identity(self.M)
		h2= np.zeros(self.M)
		G=np.concatenate((G1, G2), axis=0)
		h=np.concatenate((h1, h2), axis=None)

		P = matrix(P,tc="d")
		q = matrix(q, tc="d")
		G = matrix(G, tc="d")
		h = matrix(h, tc="d")
		# Construct the QP, invoke solver
		solvers.options['show_progress'] = False
		solvers.options['abstol']=1e-8
		solvers.options['reltol']=1e-8
		solvers.options['feastol']=1e-8
		sol = solvers.qp(P,q,G,h)
		# Extract optimal value and solution
		R=np.array(sol['x'])
		R=R.reshape(self.M,)
		opt_f=np.linalg.norm(self.Ks-R)**2/self.M
		Na=np.array(sol['z']).reshape(self.M+self.S,)
		N=Na[0:self.S]
		return R, N
	def ifunc(self,j, d):
		def integrand(z, j, d):
			return np.exp(-z**2/2)*(z+d)**j 
		return (2*np.pi)**(-.5)*quad(integrand, -d, np.inf, args = (j,d))[0]

		

