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
size = comm.Get_size()

def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi)


t1d = t_.reshape(-1, 1)
w1d = w1.reshape(-1, 1)

Cl = np.zeros((len(ell_),len(t_),len(t_)))
I0_ltc = np.squeeze(I0_ltrc,axis=2)
assert(np.squeeze(I0_ltrc,axis=2)==np.squeeze(I0_ltrc))


junksize = np.ceil(len(t_)/size)
max_num  = min((rank+1)*junksize,len(t_))
jjs      = np.arange(rank*junksize, max_num,dtype=np.int)
print(junksize,max_num)

Cl = np.zeros((len(jjs),len(ell_),len(t_)))
chimax_test = np.zeros(len(jjs))

for jj, chi1_max in enumerate((t_*chi_cmb)[jjs]):
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
      for nn in range(len(ell_)):
        Cl[jj,nn,ii] = np.real(np.sum(chifacs*I0_ltc[nn])) #not summing over r
    chimax_test[jj]=chi1_max

result = comm.gather(Cl, root=0)
ranks  = comm.gather(rank, root =0)
chimax_test  = comm.gather(chimax_test, root =0)


if rank ==0:
    cl = np.vstack([result[ii] for ii in range(size)])
    chimax_test = np.vstack([chimax_test[ii] for ii in range(size)])
    print(chimax_test)
    np.swapaxes(cl,0,1)
    print(cl.shape)
    cl*=(1./np.pi**2/2.*prefac**2)
    np.save('../G_matrices/clphipsi_parallel',cl)
