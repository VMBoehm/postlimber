#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:34:14 2019

@author: nessa
"""

from lab import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi)


t1d = t_.reshape(-1, 1)
w1d = w1.reshape(-1, 1)

Cl = np.zeros((len(ell_),len(t_),len(t_)))
I0_ltc = np.squeeze(I0_ltrc,axis=2)

for jj, chi1_max in enumerate((t_*chi_cmb)):
    for ii, chi2_max in enumerate((t_*chi_cmb)):


      chi1fac0 = (D_chi(chi2_max)*(1.+z_chi(chi2_max)))
      chi1fac0 = chi1fac0 *(chi2_max)**(1-nu_n_.reshape(1, -1))

    # no lensing kernel, because no lensing integration of chi_2
      chi2fac00 =  (D_chi(t1d*chi2_max)*(1+z_chi(t1d*chi2_max))*lensing_kernel(t1d*chi2_max, chi1_max))

      chi2fac01 =  (D_chi(1./t1d*chi2_max)*(1+z_chi(1./t1d*chi2_max))*lensing_kernel(1./t1d*chi2_max, chi1_max))

      chi2fac01 = chi2fac01 * t1d**(nu_n_.reshape(1, -1)-2)
      chi2fac0  = chi2fac00 + chi2fac01

      chifacs   = w1d*chi1fac0* chi2fac0


      result = np.zeros_like(ell_)

      Cl[:,ii,jj] = np.sum(chifacs*I0_ltc, axis=(1,2)) #not summing over r



Cl*=(1./np.pi**2/2.*prefac**2)
print('Time taken = ', time.time()-begin)

np.save('../G_matrices/clphipsi',Cl)