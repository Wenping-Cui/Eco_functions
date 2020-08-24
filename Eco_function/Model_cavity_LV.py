import time
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from Eco_function.eco_lib import *
from Eco_function.C_matrix import *
from numpy import linalg as LA
from scipy.integrate import odeint
from scipy.integrate import quad
import random as rand
from cvxopt import matrix
from cvxopt import solvers
import cvxpy as cvx
class LV_Cavity_simulation(object):
	def __init__(self, parameters):
		self.S=parameters['S']
		self.k=parameters['k']
		self.sigma_k=parameters['sigma_k']

		self.g=parameters['g']
		self.sigma_g=parameters['sigma_g']

		self.B =parameters['B']
		self.mu=parameters['mu']
		self.epsilon=parameters['epsilon']


		self.ODE_Time=parameters['ODE_Time']
		self.sample_size=parameters['sample_size']

		self.A_type=parameters['A_type']
		self.B_type=parameters['B_type']

		self.sys_gamma=parameters['sys_gamma'] # -1 asymmetric, 1 symmetry

		self.mapping=parameters['mapping_CR']
		if 	self.mapping:
			self.m=parameters['m']
			self.sigm=parameters['sigma_m']
			self.M=parameters['M']
		else:
			self.M=self.S
		


	def initialize_random_variable(self,Initial='Auto'):
		self.sim_para={}
		self.sim_para['S'] =self.S
		self.sim_para['k'] =np.random.normal(self.k, self.sigma_k, self.S) 
		self.sim_para['g']  =np.random.normal(self.g, self.sigma_g, self.S) 

		self.sim_para['N_ini']=0.1*np.ones(self.S)
		self.sim_para['ODE_Time']=self.ODE_Time
		if Initial=='Auto':
			if self.A_type=='gamma':
				self.shape=self.mu**2/(self.epsilon**2*self.M)
				self.scale=self.epsilon**2/self.mu
				self.A= np.random.gamma(self.shape, self.scale, [self.S,self.M])
			elif self.A_type=='binomial':
				self.A= np.random.binomial(1, self.epsilon, [self.S,self.M])
			elif self.A_type=='gaussian':
				self.A= np.random.normal(self.mu/self.S, self.epsilon/np.sqrt(self.S), [self.S,self.M])
			elif self.A_type=='uniform':
				self.A= np.random.uniform(0,self.epsilon, [self.S,self.M])
			# generated correlated gaussian matrixs
			if not self.mapping and self.A_type=='gaussian':
				uptriangle_ind = np.triu_indices(self.S,k=1)
				lowtriangle_ind = tuple((list(uptriangle_ind[1]), list(uptriangle_ind[0])))
				self.A1 = np.random.normal(self.mu/self.S, self.epsilon/np.sqrt(self.S), [self.S,self.M])
				self.A[uptriangle_ind] = self.sys_gamma*self.A[lowtriangle_ind] + np.sqrt(1-self.sys_gamma**2)*self.A1[uptriangle_ind]
				print("symmetry imposed", self.sys_gamma)
			elif not self.mapping and self.A_type!='gaussian':
				print("can not generate correlated matrix when it is not gaussian")

		else:
			self.A=self.C_det


		if self.mapping:
			self.sim_para['M']=self.M
			self.sim_para['g']=np.ones(self.S)
			self.K = np.random.normal(self.k, self.sigma_k, self.M) 
			self.costs= np.random.normal(self.m, self.sigm, self.S) 
			self.C = self.B + self.A
			self.sim_para['k']=self.C.dot(self.K)-self.costs
			self.sim_para['K']=self.K
			self.A= self.C.dot(self.C.T)
			self.sim_para['C']=self.C
			self.sim_para['costs']=self.costs
		else:
			self.A= self.B+self.A


		self.sim_para['A']= self.A
		return self.sim_para

	def _simulation(self,dynamics='CVXOPT',Initial='Auto'): 
		phi_N_list=[]
		phi_N_list_bar=[]
		N_list=[]
		N_list_bar=[];
		qN_list=[]
		qN_list_bar=[]
		N_survive_list=[]
		N_survive_list_bar=[]
		qN_survive_list_bar=[]
		self.N_org=[]
		Growth=[]
		nu_list=[]
		chi_list=[]
		self.lams_org =[]
		self.lams =[]
		lam_min_array=[]
		lam_max_array=[]
		self.col_N=[]
		for step in range(self.sample_size):	
			self.sim_para=self.initialize_random_variable(Initial=Initial)
			if dynamics == 'ODE':
				Model=LVModel(self.sim_para)
				T, N_f=Model._simulation()
				N = N_f[-1,:]
			elif dynamics == 'CVXOPT' and self.mapping:
				N,R=self._convex_opt()
			else:
				print("The lV model can not be mapped to CRM. The CVXOPT is not available")
			self.N_org.extend(N)
			N[N<1e-8]=0
			Growth.extend(self.sim_para['k'] -self.sim_para['A'].dot(N))
			phi_N_list.append(np.count_nonzero(N)/float(self.S));
			N_list.extend(N)
			N_list_bar.append(np.mean(N))
			qN_list_bar.append(np.mean(N**2))
			if self.mapping:
				C=self.C.copy()
				eigvs_org =np.real(LA.eigvals(C.dot(C.T)))
				C=np.delete(C, np.where(N==0),axis=0)
				A= C.dot(C.T)
				eigvs=np.real(LA.eigvals(A))
			else: 
				A=self.sim_para['A'].copy()
				eigvs_org =np.real(LA.eigvals(A))
				A=np.delete(A, np.where(N==0),axis=0)
				A=np.delete(A, np.where(N==0),axis=1)
				eigvs=np.real(LA.eigvals(A))

			if len(eigvs)>1:
				lam_min=np.amin(eigvs)
				lam_min_array.append(lam_min)
				if self.A_type== "gaussian":
					lam_max=np.amax(eigvs)
					lam_max_array.append(lam_max)
				else:
					lam_max=eigvs[np.argsort(eigvs)][-2]
					lam_max_array.append(lam_max)

			N=N[np.where(N>0)]
			self.col_N.append(N)
			self.lams_org.append(eigvs_org)
			self.lams.append(eigvs)
			N_survive_list.extend(N)
			N_survive_list_bar.append(np.mean(N))
			qN_survive_list_bar.append(np.mean(N**2))
			S=len(N);
			if np.linalg.det(A)!=0:
				if self.mapping:
					M=self.M
					Chi, Nu=self.linear_functions(R, N, C, S, M)
					nu_list.append(np.trace(np.linalg.inv(A) )/np.count_nonzero(N))
					chi_list.append(np.trace(Chi)/M)
				else:
					nu_list.append(np.trace(np.linalg.inv(A) )/np.count_nonzero(N))
					chi_list.append(0)

			else:
				print('a singular matrix appears')

		self.lam_min_array=lam_min_array
		self.mean_N, self.var_N=np.mean(N_list), np.var(N_list)
		self.mean_Grow, self.var_Grow=np.mean(Growth), np.var(Growth)
		self.mean_var_simulation={}
		self.mean_var_simulation['phi_N']=np.mean(phi_N_list)

		self.mean_var_simulation['mean_N']=self.mean_N
		self.mean_var_simulation['mean_N_s']=np.mean(N_survive_list)
		self.mean_var_simulation['mean_Growth']=self.mean_Grow
		
		self.mean_var_simulation['q_N']=self.var_N+self.mean_N**2
		self.mean_var_simulation['q_N_s']=np.var(N_survive_list)+np.mean(N_survive_list)**2
		self.mean_var_simulation['q_Growth']=self.var_Grow+self.mean_Grow**2

		self.mean_var_simulation['phi_N_bar']=np.std(phi_N_list)
		self.mean_var_simulation['mean_N_bar']=np.std(N_list_bar)
		self.mean_var_simulation['mean_N_s_bar']=np.std(N_survive_list_bar)
		self.mean_var_simulation['q_N_s_bar']=np.std(qN_survive_list_bar)
		self.mean_var_simulation['q_N_bar']=np.std(qN_list_bar)

		self.mean_var_simulation['var_N']=self.var_N
		self.mean_var_simulation['lam_min']=np.mean(lam_min_array)
		self.mean_var_simulation['lam_min_bar']=np.std(lam_min_array)
		self.mean_var_simulation['lam_max']=np.mean(lam_max_array)
		self.mean_var_simulation['lam_max_bar']=np.std(lam_max_array)
		self.mean_var_simulation['nu']=np.mean(nu_list)
		self.mean_var_simulation['chi']=np.mean(chi_list)

		self.N_survive_List=N_survive_list
		self.phin_list=phi_N_list
		self.N_List=N_list
		self.G_List=Growth
		return self.mean_var_simulation

	def _convex_opt(self,Initial='Auto'):
		# Define QP parameters (directly)
		M = np.identity(self.M)
		P = np.dot(M.T, M)
		q = -np.dot(self.sim_para['K'],M).reshape((self.M,))

		G= self.sim_para['C']
		h= self.sim_para['costs']


		P = matrix(P,tc="d")
		q = matrix(q, tc="d")
		G = matrix(G, tc="d")
		h = matrix(h, tc="d")
		# Construct the QP, invoke solver
		solvers.options['show_progress'] = False
		solvers.options['abstol']=1e-16
		solvers.options['reltol']=1e-16
		solvers.options['feastol']=1e-16
		sol = solvers.qp(P,q,G,h)
		# Extract optimal value and solution
		R=np.array(sol['x'])
		R=R.reshape(self.M,)
		opt_f=np.linalg.norm(self.sim_para['K']-R)**2/self.M
		Na=np.array(sol['z']).reshape(self.sim_para['S'],)
		N=Na[0:self.S]
		return N, R
	def linear_functions(self,R, N, C, S, M):
		A=np.concatenate((np.concatenate((C, np.eye(M))),np.concatenate((np.zeros([S,S]), C.T))),axis=1)
		B=np.eye(M+S)
		Sup=np.linalg.inv(A).dot(B)
		Chi_R=Sup[0:M,S:]
		Nu_N=Sup[M:, 0:S]
		return Chi_R, Nu_N
	


##########################################################################
class LVModel(object):
    def __init__(self, par):
        '''initilize  parameters'''
        self.S = par['S']  # system size S speces
        self.A = par['A']    #  species interactions
        self.k = par['k']    #  carrying capacities
        self.g = par['g']    #  intrinsic growth rates

        self.t0, self.t1, self.Nt = par['ODE_Time'] # ODE simulation time
        self.N_ini = par['N_ini']   # initialize species abundance

        '''Assert initilization error'''
        assert (self.S  == len(self.N_ini)),"initialed species abundance size error"
        assert (self.S  == len(self.k)), "carrying capacities k size error"
        assert (self.S  == len(self.g)), "intrinsic growth rates g size error"
        assert (self.A.ndim == 2), "The interaction matrix A must be a square matrix"
        assert (np.shape(self.A)[0] == np.shape(self.A)[1] and np.shape(self.A)[0]==self.S), "The interaction matrix A must be a square matrix"

        self.N_f = np.zeros(self.S)  # finial species abundance

    def _simulation(self,):
        N_ini=self.N_ini
        T = np.linspace(self.t0, self.t1, num=self.Nt)
        par = [self.A, self.k, self.g] 
        N = odeint(self._LV_dynamics, N_ini, T, args=(par,),mxstep=50000, atol=10 ** -7, hmin=1e-20)
        N[np.where(N < 10 ** -4)] = 0

        self.N_f = N[-1, :]
        self.survive = np.count_nonzero(self.N_f)   

        return T,N

    def _LV_dynamics(self, N, t, par):
        [A , k, g] = par;
        N[np.where(N < 0)] = 0;
        output_vector = N * g * (k - A.dot(N)) 
        return output_vector
		



