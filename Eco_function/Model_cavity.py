import time
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from Eco_function.eco_lib import *
from Eco_function.eco_func import *
from Eco_function.C_matrix import *
from numpy import linalg as LA
from scipy.integrate import odeint
from scipy.integrate import quad
import random as rand
from cvxopt import matrix
from cvxopt import solvers
import cvxpy as cvx
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
		self.Metabolic_Tradeoff=False
		self.flag_crossfeeding=False
		self.C_type='gaussian'
		self.B_type='null'
		self.gamma_flag='S/M'
		self.D=0
		self.Bnormal=False
		self.non_zero_resource=range(self.M)
		self.p_c=0.2
		self.epsilon=10**(-3)
	def initialize_random_variable(self,):
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
		self.t0 = 0;
		self.t1 = self.parameters['t1'];
		self.Nt = self.parameters['Nt']
		self.T_par = [self.t0, self.t1, self.Nt];

		if self.Metabolic_Tradeoff:
			self.costs=np.sum(self.C, axis=1)+self.epsilon*np.random.normal(0, 1, self.S)
		else:
			self.costs=np.random.normal(self.cost, self.sigma_m, self.S)		#Ode solver parameter

########################################################################################################
		###   Make the determined matrix
########################################################################################################
		B=0
		if self.B_type=='identity':
			B=np.identity(self.M)
		elif self.B_type=='circulant':
			D = [7, 1]  # generalist, specialist
			B=circ(self.M, D[1])
		elif self.B_type=='block':
			B= block(int(self.M/10), 10)
########################################################################################################
		if self.Bnormal:
				B=B/np.sum(B[0,:])
##################################################################################################################						
		if self.gamma_flag=='S/M':
			self.C=B+np.random.normal(self.mu/self.M, self.sigma_c/np.sqrt(self.M), [self.S,self.M])
		if self.gamma_flag=='M/S':
			self.C=B+np.random.normal(self.mu/self.S, self.sigma_c/np.sqrt(self.S), [self.S,self.M])

		if self.C_type=='gamma':
			self.shape=self.mu**2/(self.epsilon**2*self.M)
			self.scale=self.epsilon**2/self.mu
			self.C= B+np.random.gamma(self.shape, self.scale, [self.S,self.M])
		elif self.C_type=='binomial':
			self.C= B+np.random.binomial(1, self.p_c, [self.S,self.M])
		elif self.C_type=='gaussian':
			self.C= B+np.random.normal(self.mu/self.S, self.epsilon/np.sqrt(self.S), [self.S,self.M])
		elif self.C_type=='uniform':
			self.C= B+np.random.uniform(0,self.epsilon, [self.S,self.M])
		#shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
		#self.C= np.random.gamma(shape, scale, [self.S,self.M])
		self.R_ini=0.1*np.ones(self.M)
		self.N_ini=0.1*np.ones(self.S)
		self.sim_pars = [self.flag_crossfeeding, self.M, self.S, self.R_ini, self.N_ini,self.T_par, self.C, self.energies, self.tau_inv, self.costs, self.growth, self.Ks] 
		if self.flag_crossfeeding: 
			self.sim_pars = [self.flag_crossfeeding, self.M, self.S, self.R_ini, self.N_ini, [self.t0, self.t1, self.Nt], self.C, self.energies, self.tau_inv,self.costs,self.growth, self.D, self.non_zero_resource,self.Ks]
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
		Chi_array=[];
		Nu_array=[];
		Lamc_array=[]
		lam_min_array=[]
		lam_min_cor_array=[]
		lam_min_ran_array=[]
		Lam_array=[];
		self.sev = np.array([])
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
					Model.flag_linear=True;
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
			if Simulation_type=='QP' and  Dynamics=='quadratic':
				R, N=self.Quadratic_programming(self,)
				R[np.where(R < 10 ** -6)] = 0
				N[np.where(N < 10 ** -6)] = 0
				Model_costs_power=N.dot(self.costs)
				Model_survive=np.count_nonzero(N)
			if Simulation_type=='CVXOPT' and Dynamics=='linear':
				self.Ks[np.where(self.Ks<0)]=0;
				R, N,opt_v=self.CVXOPT_programming(self,)
				R[np.where(R < 10 ** -6)] = 0
				N[np.where(N < 10 ** -6)] = 0
				Model_costs_power=N.dot(self.costs)
				Model_survive=np.count_nonzero(N)
				Opti_f.append(opt_v)
			if Dynamics=='quadratic':	
				Opti_f.append((np.linalg.norm(self.Ks-R))**2/self.M)
			self.R_f, self.N_f=R,N
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
			C=self.C;
			C=np.delete(C, np.where(R==0),axis=1)
			C=np.delete(C, np.where(N==0),axis=0)
			eigvs=np.real(LA.eigvals(np.dot(C,C.transpose())))
			eigvs_cor=np.real(LA.eigvals(np.einsum('i,ij->ij', N[np.where(N>0)], np.dot(C,C.transpose()))))
			N_bar=np.random.permutation(N[np.where(N>0)])
			eigvs_ran=np.real(LA.eigvals(np.einsum('i,ij->ij', N_bar, np.dot(C,C.transpose()))))
			if len(eigvs)>1:
				Lam_array.extend(eigvs_cor)
				lam_min=np.amin(eigvs)
				lam_min_array.append(lam_min)
			else:
				lam_min_array.append(0)
			if len(eigvs_cor)>1:
				Lamc_array.extend(eigvs_cor)
				lam_min_cor=np.amin(eigvs_cor)
				lam_min_cor_array.append(lam_min_cor)
			else:
				lam_min_cor_array.append(0)
			if len(eigvs_ran)>1:
				lam_min_ran=np.amin(eigvs_ran)
				lam_min_ran_array.append(lam_min_ran)
			else:
				lam_min_ran_array.append(0)
			R=R[np.where(R>0)]
			N=N[np.where(N>0)]
			N_survive_list.extend(N)
			S=len(N);
			M=len(R);
			if Dynamics=='linear':
				JR=np.concatenate((-np.diag(np.einsum('i,ij', N, C))-np.diag(R),-np.dot(np.diag(R), C.T)), axis=1);
				JN=np.concatenate((np.dot(np.diag(N), C),np.zeros([ S, S])), axis=1);
				J_all=np.concatenate((JR, JN), axis=0);
				ev,_ = np.linalg.eig(J_all)
			if Dynamics=='quadratic':
				JR=np.concatenate((-np.diag(R),-np.dot(np.diag(R),C.T)), axis=1);
				JN=np.concatenate((np.dot(np.diag(N),C),np.zeros([S, S])), axis=1);
				J_all=np.concatenate((JR, JN), axis=0);
				ev,_ = np.linalg.eig(J_all)
				A=np.concatenate((np.concatenate((C, np.eye(M))),np.concatenate((np.zeros([S,S]), C.T))),axis=1);
				if np.linalg.det(A)==0: continue;
				self.A=A
				self.M_p=M
				self.S_p=S
				Chi_R, Nu_N=self.linear_response_q(A, S, M)
				chi=np.trace(Chi_R)/self.M
				nu=np.trace(Nu_N)/self.S
				Chi_array.append(chi);
				Nu_array.append(nu);
			self.sev = np.append(self.sev, ev)
		self.packing=np.asarray(phi_N_list)/np.asarray(phi_R_list)
		self.lamcs=Lamc_array
		self.lam_min_cor_array=lam_min_cor_array
		self.lams=Lam_array
		self.lam_min_array=lam_min_array
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
		self.mean_var_simulation['lam_min']=np.mean(lam_min_array)
		self.mean_var_simulation['lam_min_cor']=np.mean(lam_min_cor_array)
		self.mean_var_simulation['lam_min_ran']=np.mean(lam_min_ran_array)
		if Dynamics=='quadratic':
			Nu_array=np.asarray(Nu_array)
			self.mean_var_simulation['nu']=np.mean(Nu_array)
			self.mean_var_simulation['nu_threshold']=np.mean(Nu_array[np.where((Nu_array>-10000) & (Nu_array<0))])
			self.mean_var_simulation['chi']=np.mean(Chi_array)
		else:
			self.mean_var_simulation['nu']='NaN'
			self.mean_var_simulation['chi']='NaN'
			self.mean_var_simulation['nu_threshold']='NaN'
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
	
	def Quadratic_programming(self, Initial='Auto'):
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


	def CVXOPT_programming(self, Initial='Auto'):
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

		R = cvx.Variable(shape=(self.M,1))
		K = self.Ks.reshape((self.M,1))
		h=h.reshape((self.M+self.S,1))

		# Construct the QP, invoke solver
		obj = cvx.Minimize(cvx.sum(cvx.kl_div(K+1e-10, R+ 1e-10)))
		constraints =[G*R <= h]
		prob = cvx.Problem(obj, constraints)
		prob.solver_stats
		prob.solve(solver=cvx.ECOS,abstol=1e-8,reltol=1e-8,warm_start=True)
		# Extract optimal value and solution
		N=(constraints[0].dual_value)[0:self.S]
		R=R.value
		R=R.reshape(self.M);
		N=N.reshape(self.S);
		return R, N, prob.value
	def ifunc(self,j, d):
		def integrand(z, j, d):
			return np.exp(-z**2/2)*(z+d)**j 
		return (2*np.pi)**(-.5)*quad(integrand, -d, np.inf, args = (j,d))[0]

	def linear_response_q(self, A, S_p, M_p):
		Chi=np.zeros([M_p,M_p+S_p])
		for i in range(M_p):
			b = np.zeros(S_p+M_p)
			b[S_p+i]=1
			Chi[i,:] = np.linalg.solve(A, b)
		Chi_R=Chi[:, 0:M_p]
		Chi_N=Chi[:, M_p:]
		Nu=np.zeros([S_p,M_p+S_p])
		for i in range(S_p):
				b = np.zeros(S_p+M_p)
				b[i]=1
				Nu[i,:] = np.linalg.solve(A, b)
		Nu_R=Nu[:, 0:M_p]
		Nu_N=Nu[:, M_p:]
		return Chi_R, Nu_N

		

