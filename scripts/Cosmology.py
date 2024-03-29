# -*- coding: utf-8 -*-
"""
Created on 12.02.2015

@author: Vanessa Boehm

Cosmology.py:
	* Cosmological Parameter sets
	* Class Cosmology()
	* Class CosmoData()
"""

from __future__ import division
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import splrep, splev
from scipy.integrate import odeint
import matplotlib.pyplot as pl
import copy

import Constants as const
#import HelperFunctions as HF
from classy import Class
import pickle

EmasCosmology=[{'name':'OurPostBornPaper'},
{"A_s": 2.10732e-09,
"h": 0.677,
"k_pivot": 0.05,
"n_s": 0.96824,
"omega_b": 0.02247,
"omega_cdm": 0.11923
}]

""" Planck 2013 Cosmological parameters from CMB and CMB temperature lensing, no neutrinos """
Planck2013_TempLensCombined=[{
'name':"Planck2013_TempLensCombined"},{
'h': 0.6714,
'omega_b' : 0.022242,
'omega_cdm': 0.11805,
'Omega_k' : 0.0,
'tau_reio' : 0.0949,
'A_s'    : 2.215*1e-9,
'n_s'    : 0.9675,
'k_pivot' : 0.05}]

Planck2013_Giulio=[{
'name':"Giulio_PB"},{
'h': 0.68,
'omega_b' : 0.02312,
'omega_cdm': 0.124848,
'Omega_k' : 0.0,
'tau_reio' : 0.0949,
'A_s'    : 2.06*1e-9,
'n_s'    : 0.96,
'k_pivot' : 0.05}]

ToshiyaComparison=[{
'name':"Toshiya"},{
'h': 0.6751,
'omega_b':0.0223,
'omega_cdm':0.119,
'Omega_k':0.0,
'tau_reio':0.063,
'A_s'   :2.13e-9,
'n_s'   :0.965,
'k_pivot' : 0.05,

}]

Jia=[{
'name':"JiaGalaxyLens"},{
'h': 0.7,
'omega_b':0.0223,
'omega_cdm':0.12470,
'Omega_k':0.0,
'A_s'   :2.1000e-9,
'n_s'   :0.97,
'k_pivot' : 0.05
}]


#simple settings for plot
JiaNu={
'omega_cdm': 0.12362,
'N_ur':0.00641,
'N_ncdm' : 3,
'm_ncdm' : str(0.02188)+','+str(0.02352)+','+str(0.05519),
'ncdm_fluid_approximation': 1,
'deg_ncdm' : '1.0, 1.0, 1.0',
'k_max_tau0_over_l_max': 3.0,
'k_min_tau0': 0.002,
'k_pivot': 0.05,
'k_step_sub': 0.015,
'k_step_super': 0.0001,
'k_step_super_reduction': 0.1,
'ncdm_fluid_approximation': 1,
'ncdm_fluid_trigger_tau_over_tau_k':51.,
}

# getting 9965 difference with this
#{'A_s': 2.1e-09,
# 'N_ncdm': 3,
# 'N_ur': 0.00641,
# 'Omega_k': 0.0,
# 'P_k_max_1/Mpc': 10.0,
# 'h': 0.7,
# 'k_max_tau0_over_l_max': 3.0,
# 'k_min_tau0': 0.002,
# 'k_pivot': 0.05,
# 'k_step_sub': 0.015,
# 'k_step_super': 0.0001,
# 'k_step_super_reduction': 0.1,
# 'm_ncdm': '0.021880597824,0.02352325212,0.055187350056',
# 'n_s': 0.97,
# 'ncdm_fluid_approximation': 1,
# 'ncdm_fluid_trigger_tau_over_tau_k':51.,
# 'omega_b': 0.0223,
# 'omega_cdm': 0.12362,
# 'output': 'tCl mPk'}


Namikawa=[{
'name':"Namikawa_Paper"},{
'h': 0.6712,
'omega_b':0.0223,
'omega_cdm':0.119,
'Omega_k':0.0,
'tau_reio':0.0630,
'A_s'   :2.13e-9,
'n_s'   :0.965,
'N_ncdm': 2,
'N_ur':1.0196,
#'m_ncdm': "0.05, 0.01",
'k_pivot' : 0.05,
'tau_reio':0.0630
#'ncdm_fluid_approximation': 2,
#'ncdm_fluid_trigger_tau_over_tau_k':51.,
#'tol_ncdm_synchronous':1.e-10,
#'tol_ncdm_bg':1.e-10,
#'l_max_ncdm':51
}]

Pratten=[{
'name':"Pratten_Paper"},{
'h': 0.67,
'omega_b':0.022,
'omega_cdm':0.123,
'Omega_k':0.0,
'A_s'   :2.0*1e-9,
'n_s'   :0.965,
'k_pivot':0.05}]


""" Planck 2015 Cosmological parameters from different combinations of constraints"""
Planck2015_TTlowPlensing=[{
'name':"Planck2015_TTlowPlensing"},{
'h': 0.6781,
'omega_b' : 0.02226,
'omega_cdm': 0.1186,
'Omega_k' : 0.0,
#'tau_reio' : 0.066,
'ln10^{10}A_s':3.062,
'n_s'    : 0.9677,
'k_pivot' : 0.05}]


""" Planck 2015 Cosmological parameters from different combinations of constraints"""
Planck2015=[{
'name':"Planck2015"},{
'h': 0.68,
'omega_b' : 0.0223,
'omega_cdm': 0.119,
'Omega_k' : 0.0,
#'tau_reio' : 0.066,
'ln10^{10}A_s':3.062,#sigma8:  0.829472350106
'n_s'    : 0.97,
'k_pivot' : 0.05}]
#print derived sigma_8

LimberTest=[{'name':'postLimberTest'},{
'k_pivot' : 0.05,
'A_s' : 2.137e-9,
'n_s' : 0.97,
'h' : 0.68,
'omega_b' : 0.0223,
'N_ur' : 3.046,
'omega_cdm' : 0.119}]

Planck2015_TTTEEElowPlensing=[{
'name':"Planck2015_TTTEEElowPlensing"},{
'h': 0.6751,
'omega_b' : 0.02226,
'omega_cdm': 0.1193,
'Omega_k' : 0.0,
'tau_reio' : 0.063,
'ln10^{10}A_s': 3.059,
'n_s'    : 0.9653}]

""" Jia's and Colin's simulation """
SimulationCosmology=[{'name':"Jias_Simulation"},{
'T_cmb':2.725,
'omega_cdm':0.129600,
'omega_b':0.023846,
'h':0.720,
'n_s':0.960,
'A_s':1.9715*1e-9,
'Omega_k':0.0,
'tau_reio' : 0.087,
'k_pivot' : 0.002,
'N_eff': 3.046,
'YHe' : 0.24,
'N_ncdm' : 0,
'halofit_k_per_decade' : 3000.,
'accurate_lensing':1}]


""" Takada & Jain paper """
Takada=[{'name':"Takada"},{
'T_cmb':2.725,
'omega_cdm':0.3*0.72**2,
'omega_b':0.05*0.72**2,
'h':0.720,
'n_s':1.,
'A_s':1.84*1e-9,
'Omega_k':0.0}]


#corresponding to pk_ref
acc_1={
"k_min_tau0":0.002,
"k_max_tau0_over_l_max":5.,
"k_step_sub":0.015,
"k_step_super":0.0001,
"k_step_super_reduction":0.1,
'k_per_decade_for_pk': 20,
'perturb_sampling_stepsize':0.01,
'tol_perturb_integration':1.e-6,
'halofit_k_per_decade': 3000.
}


""" Gil-Marin et al Simulations """
BispectrumSimulations=[{'name':"Gil-Marin_et_al"},{
'T_cmb':2.725,
'omega_cdm':0.27*0.7**2-0.023,
'omega_b':0.023,
'h':0.7,
'n_s':0.95,
'A_s':2.585*1e-9,
'Omega_k':0.0,
'k_pivot' : 0.002}]

MatterOnly=[{'name':"Matter_Only"},{
'T_cmb':2.725,
'omega_cdm':0.5084,
'omega_b':0.01,
'h':0.720,
'n_s':0.960,
'A_s':1.971*1e-9,
'Omega_k':0.0,
'k_pivot' : 0.002}]



class CosmoData():
	""" class of z-dependent quantities (distances etc.) for a given cosmology """

	def __init__(self, params, z, test=False):
		"""computes H(z), comoving distance, scale factor, prefactor in poisson equation as function of z
			* cosmo: 	instance of class Cosmology (paramter s for CLASS)
			* z: 	     	array of redshifts
		"""
		print("computing distances, derived parameters...")

		self.class_params     = params
		self.z                = z

		closmo                = Class()
		closmo.set(self.class_params)

		closmo.compute()

		cosmo_b               = closmo.get_background()

		class_z               = cosmo_b['z'][::-1]
		class_chi             = cosmo_b['comov. dist.'][::-1]

		class_D               = cosmo_b['gr.fac. D'][::-1]/cosmo_b['gr.fac. D'][-1] #normalized to todays value

		#for growth func
		self.w0_fld 		      = -1.
		self.wa_fld 		      = 0.

		LJ_D,zD		 	      = self.get_growth(closmo,np.linspace(0.,max(z),100))

		self.LJ_D_z           = ius(zD,LJ_D)

		self.D_chi            = ius(class_chi,class_D)
		self.D_z              = ius(class_z,class_D)


		self.chi              = ius(class_z,class_chi)
		self.zchi             = ius(class_chi,class_z)

		derivParams           = closmo.get_current_derived_parameters(['z_rec'])

		self.z_cmb            = derivParams['z_rec']

		self.chi_cmb          = self.chi(self.z_cmb)

		self.a                = self.get_a(z)

		# CLASS units: c=1, all quantities in Mpc^n
		self.H_0              = cosmo_b['H [1/Mpc]'][-1]*const.LIGHT_SPEED

		class_H               = cosmo_b['H [1/Mpc]'][::-1]*const.LIGHT_SPEED

		self.H                = ius(class_z,class_H)

		self.Omega_m0         = 1.-cosmo_b['(.)rho_lambda'][-1]/(cosmo_b['(.)rho_crit'][-1])

		self.lens_prefac       = self.Poisson_factor()

		self.closmo       = closmo

		print(closmo.get_current_derived_parameters(['Neff']))
		print(closmo.get_current_derived_parameters(['h']))
		print(closmo.get_current_derived_parameters(['m_ncdm_tot']))
		print(closmo.get_current_derived_parameters(['omega_m']))

		closmo.struct_cleanup()
		closmo.empty()

		## Check if chi interpolation works and if matter power spectrum makes sense
		if test:
			#should be the same, if interpolation works correctly
			pl.figure()
			pl.plot(self.z,self.chi(self.z), label="class interpolated", marker="o")
			pl.plot(class_z[::-1],class_chi[::-1], label="class", color="r", ls="--")
			pl.xlim(min(self.z),max(self.z))
			pl.xlabel("z")
			pl.ylabel("Com. Distance [Mpc]")
			pl.legend()
			pl.show()

			test_params=copy.deepcopy(self.class_params)
			k_aux=np.exp(np.linspace(np.log(0.004*0.72),np.log(1.),200))
			test_params['z_max_pk']=10.
			#Maximum k value in matter power spectrum
			test_params['P_k_max_1/Mpc'] = max(k_aux)
			test_params['k_min_tau0'] = min(k_aux*13000.)
			test_params['output']= 'mPk'
			test_params['tol_perturb_integration']=1.e-6
			#test_params['non linear']='halofit'

			closmo_test = Class()
			closmo_test.set(test_params)

			closmo_test.compute()


			test_zs=np.array([0.,0.6,1.,1.5,10.])
			P=np.empty((len(test_zs),len(k_aux)))
			for ii in range(len(test_zs)):
				P[ii]=[closmo_test.pk(k,test_zs[ii]) for k in k_aux]
			print(P[ii].shape)

			# this plot can be compared with literature
			pl.figure()
			pl.loglog(k_aux/(self.H_0/100.),np.array(P[0])*(self.H_0/100.)**3)
			pl.xlabel(r'$k[h/Mpc]$')
			pl.ylabel(r'$P[Mpc/h]^3$')
			pl.show()

			#check if growth function is correct
			pl.figure()
			ii=0
			for zi in test_zs:
				pl.plot(k_aux,[self.D_z(zi)**2]*len(k_aux), label=r'$D(z=%.1f)^2$'%zi, ls=":")

				pl.plot(k_aux,[self.LJ_D_z(zi)**2]*len(k_aux), label=r'$D(z=%.1f)^2$'%zi, ls="--")
				pl.plot(k_aux,np.array(P[ii])/np.array(P[0]),label=r'$P(z=%d)/P(0)$'%zi)
				ii+=1
			pl.ylim(1e-4,1.)
			pl.xlim(min(k_aux),max(k_aux))
			pl.xlabel(r'$k[h/Mpc]$')
			pl.legend()
			pl.show()

			closmo_test.struct_cleanup()

			closmo_test.empty()


	def get_a(self,z):
		""" converts a to z """
		a=1./(1.+z)
		return a

	def get_z(self,a):
		""" converts z to a """
		z=(1./a)-1.
		return z

	def Poisson_factor(self):
		""" computes the proportionality constant of the Poisson equation """

		alpha= 1.5*self.H_0**2.*self.Omega_m0/(const.LIGHT_SPEED**2)

		return alpha

	def dchidz(self,z):
		""" dDcom/dz """

		result = const.LIGHT_SPEED/self.H(z)

		return result

	def dzdchi(self,z):
		""" dDcom/dz """

		result = self.H(z)/const.LIGHT_SPEED

		return result

	def get_Cls(self,tag,nl=False,lmax=6000,path='./outputs/ClassCls/'):
		params =copy.deepcopy(self.class_params)
		params['output']='tCl lCl pCl mPk'
		params['lensing']='yes'
		if nl:
			params['non linear']="halofit"
			tag+="_nl"
		else:
			params['non linear']=" "
		params['tol_perturb_integration']=1e-6
		params['perturb_sampling_stepsize']=0.01
		params['k_min_tau0']=0.002
		params['k_max_tau0_over_l_max']=10.
		params['l_max_scalars']=lmax+2000
		params['halofit_k_per_decade']=3000


		closmo 					 = Class()
		closmo.set(params)

		print("Calculalating Cls... with settings",self.class_params)

		closmo.compute()
		print('sigma8: ',closmo.sigma8())


		cl_len=closmo.lensed_cl(lmax)

		cl_unl=closmo.raw_cl(lmax)

		pickle.dump([params,cl_unl,cl_len],open(path+'class_cls_%s.pkl'%tag,'wb'))

		print('dumped to ', path+'class_cls_%s.pkl'%tag)

		return True


	def get_linPm(self,k_array,test=True,get_n=False,z_max=1.5,z_=None):

		params=copy.deepcopy(self.class_params)

		params['output']='tCl mPk'


		params['P_k_max_1/Mpc']= max(k_array)
		params['z_max_pk']     = z_max

		params['non linear']=""
		params.update(acc_1)

		print(min(k_array),max(k_array))
		print(params)
		closmo = Class()
		closmo.set(params)
		closmo.compute()

		P=[closmo.pk(k,0.) for k in k_array]

		sigma8 		= closmo.sigma8()

		self.k_NL 	= []

		print(min(k_array))
		print(max(k_array))

		k_i=np.exp(np.linspace(np.log(min(k_array)),np.log(max(k_array)),1000))#1./self.cosmo.class_params['h']
		for z in z_[np.where(z_<=z_max)]:
			Pk = np.asarray([closmo.pk(k,z) for k in k_i])
			self.k_NL+=[min(k_i[np.where(Pk*k_i**3/(2*np.pi**2)>1.)])]

		pl.figure()
		pl.semilogy(z_,self.k_NL)
		pl.xlabel('z')
		pl.ylabel('$k_{NL}$')
		pl.show()
		print("sigma8:", sigma8)

		self.sigma8_z 		= splrep(z_,sigma8*(self.LJ_D_z(z_))) #sigma_8 today rescaled to other redshifts
		pl.figure()

		pl.plot(z_,splev(z_,self.sigma8_z))
		pl.xlabel('z')
		pl.ylabel('$\sigma_8$')
		pl.show()


		self.n = None#iddget_n:
      #ass
			#self.n   = HF.get_derivative(np.log(k_array),np.log(P),method="spl")

		h   = self.class_params['h']
		k_  = np.exp(np.linspace(np.log(1e-3*h),np.log(10*h),100))
		pl.figure()
		for z_ in [0.,1.,z_max]:
				plk=[closmo.pk(kk,z_)for kk in k_]
				pl.loglog(k_/h,np.asarray(plk)*h**3,label='z=%f'%z_)
		pl.xlabel(r'$k [h/Mpc]$')
		pl.xlim(min(k_/h),max(k_/h))
		pl.ylabel(r'$P(k) [Mpc/h]^3$')
		pl.ylim(0.1,100000)
		pl.legend(loc='best')
		pl.savefig('pow_spec_lin.png')



		if test:
			#plots D_+/a, compare e.g. Cosmology script Matthias
			pl.figure()
			pl.plot((self.z+1)**(-1),self.D_z(self.z), label="class interpolated", marker="o")
			pl.plot((self.z+1)**(-1),(1.+self.z)**(-1), label="a", marker="o")
			pl.xlim((min(self.z)+1)**(-1),(max(self.z)+1)**(-1))
			pl.xlabel("a")
			pl.ylabel("Growth Function D_+")
			pl.legend()
			pl.show()

			if get_n:
				pl.figure()
				pl.semilogx(k_array,self.n(np.log(k_array)),marker="o")
				pl.xlim(min(k_array),max(k_array))
				pl.xlabel("k")
				pl.ylabel("spectra index n")
				pl.show()

		closmo.struct_cleanup()
		closmo.empty()

	def get_abc(self,k,z,z_max,fit_type='GM'):


		if fit_type=='GM':
			a1 = 0.484
			a4 = 0.392
			a7 = 0.128
			a2 = 3.740
			a5 = 1.013
			a8 = -0.722
			a3 = -0.849
			a6 = -0.575
			a9 = -0.926
		if fit_type=='SC':
			a1 = 0.25
			a4 = 1.
			a7 = 1.
			a2 = 3.5
			a5 = 2.
			a8 = 0.
			a3 = 2.
			a6 = -0.2
			a9 = 0.

		try:
			self.n
		except:
			self.get_linPm(k,test=True,get_n=True,z_max=z_max,z_=z)

		n=self.n(np.log(k))


		Q=(4.-2.**n)/(1.+2**(n+1.))


		self.a_nk=[]
		self.b_nk=[]
		self.c_nk=[]
		j=0
		#checked
		pl.figure()
		for z_ in z[np.where(z<=z_max)]:
			q=k/self.k_NL[j]
			a_nk_z=(1.+splev(z_,self.sigma8_z)**a6*np.sqrt(0.7*Q)*(a1*q)**(n+a2))/(1.+(q*a1)**(n+a2))
			pl.plot(k,a_nk_z)
			self.a_nk+=[splrep(k,a_nk_z)]
			b_nk_z=(1.+0.2*a3*(n+3.)*(q*a7)**(n+3.+a8))/(1.+(q*a7)**(n+3.5+a8))
			c_nk_z=(1.+4.5*a4/(1.5+(n+3.)**4)*(q*a5)**(n+3.+a9))/(1.+(q*a5)**(n+3.5+a9))
			pl.plot(k,b_nk_z)
			pl.plot(k,c_nk_z)
			self.b_nk+=[splrep(k,b_nk_z)]
			self.c_nk+=[splrep(k,c_nk_z)]
			j+=1
#		self.b_nk=splrep(k,b_nk)
#		self.c_nk=splrep(k,c_nk)

		k_test=np.exp(np.linspace(np.log(min(k)),np.log(max(k)),200))

		pl.semilogx(k_test,splev(k_test,self.b_nk[0]),ls="--")
		pl.plot(k_test,splev(k_test,self.c_nk[0]),ls="--")
		pl.plot(k_test,splev(k_test,self.a_nk[0]),ls="--")

		pl.savefig("bn_cn_test.png")
#
###### Steffens growth factor version, needed if no cosmological constant, needs to be tested#########
	def Omega_m(self,a,closmo):
		z=self.get_z(a)
		result = ( closmo.Omega_m()*(1.+z)**3 * (closmo.Hubble(0)/closmo.Hubble(z))**2 )
		return result

	def w_de(self, a, closmo):
		result = self.w0_fld + (1. - a) * self.wa_fld
		return result

    #defined in Linder+Jenkins MNRAS 346, 573-583 (2003)
    #solved integral by assuming linear w_de scaling analytically
	def x_plus(self,a,closmo):
		aux = 3.0 * self.wa_fld * (1. - a)
		result = closmo.Omega_m() / (1. - closmo.Omega_m()) * a**(3. * (self.w0_fld + self.wa_fld)) * np.exp(aux)
		return result
#
# growth function D/a from Linder+Jenkins MNRAS 346, 573-583 (2003)
# independent of a at early times: initial conditions g(a_early) = 1, g'(a_early) = 0
# choose a_early ~ 1./30. where Omega_m ~ 1
	def g(self,y,a,closmo):
		y0 = y[0]
		y1 = y[1]
		y2 = -(7./2. - 3./2. * self.w_de(a,closmo)/(1+self.x_plus(a,closmo))) * y1 / a - 3./2. * (1-self.w_de(a,closmo))/(1.+self.x_plus(a,closmo)) * y0 / a**2
		return y1, y2

	def get_growth(self,closmo,z_array):

#        if self.class_setup==False:
#            self.closmo = Class()
#            self.cosmo.class_params['P_k_max_1/Mpc'] = 15.
#            self.closmo.set(self.cosmo.class_params)
#
#            self.closmo.compute()
#            self.class_setup=True

#        if (z_array.all()==self.z.all()):
#            z_array=self.z[np.where(self.Omega_z<0.9999)]

		a_array=self.get_a(z_array[::-1])

		init = a_array[0], 1.

		solution = odeint(self.g, init, a_array, args=(closmo,))

		growspline = a_array*solution[:,0]/solution[-1,0]

		return growspline[::-1], z_array
##### Steffens version #########
