##Save the files for the Cl_phi_delta(r2d*t2d*chi_cmb, r2d*chi_cmb)

import numpy as np
from mpi4py import MPI
import os, sys
import json
from scipy.interpolate import interp1d
from scipy.integrate import simps, quadrature
from time import time

#My imports
from lab import *
import tools
from getcl import getcl

#MPI
comm = MPI.COMM_WORLD
rank, wsize = comm.rank, comm.size
print(wsize, rank)


#Kernels
def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) #* (1.+z_chi(xi))

galaxy_kernel = lambda xi, xmax : lsst_kernel_cb(xi)


In_ltrc = [I0_ltrc, None, I2_ltrc, None, I4_ltrc]

r1d, t1d = t_.reshape(1, -1), t_.reshape(-1, 1)
w11, w12 = w1.reshape(1, -1), w1.reshape(-1, 1)
grid = r1d*t1d*chi_cmb
# inflate by one dimensions (nu_n)
r2d, t2d = np.expand_dims(r1d, 2), np.expand_dims(t1d, 2)
w11, w12 = np.expand_dims(w11, 2), np.expand_dims(w12, 2)

####################


#t_ = t_[:10]

chi1maxs = t_*chi_cmb
chi1maxsplit = np.array_split(chi1maxs, wsize)
chiindex = np.arange(t_.size)
indexsplit = np.array_split(chiindex, wsize)

maxload = max(np.array([len(i) for i in indexsplit]))
if rank == 0: print('Maxload = ', maxload)
if rank == 0: print('chi1 split amongst ranks = ', chi1maxsplit)

nushift = 2
prefindex = 1

I_ltrc = In_ltrc[nushift]
I_ltc = np.squeeze(I_ltrc)

kernel2 = lensing_kernel

t1d = t_.reshape(-1, 1)
w1d = w1.reshape(-1, 1)

for index in indexsplit[rank]:

    print('Rank %d for index '%rank, index)
    chi1 = chi1maxs[index]
    
    begin=time()

    #if rank == 0: print(ichi1, chi1)
    chi1fac = chi1**(1-(nushift + nu_n_.reshape(1, -1)))
    chi1fac *= D_chi(chi1)

    tosave = np.zeros((ell_.size, t_.size))
    for ichi1, chi2max in enumerate(t_*chi1):

        chi2fac = (kernel2(t1d*chi1, chi2max) * D_chi(t1d*chi1))

        Cl = np.zeros_like(ell_)
        for ii in range(ell_.size):
            matrix = w1d * chi2fac* chi1fac * I_ltc[ii]
            Cl[ii] = np.sum(matrix)
        #Cl = (w1.reshape(-1, 1)*chi1fac* chi2fac * I_ltc).sum(axis=(1, 2)) 
        Cl = 2 * Cl *1./np.pi**2/2.* prefac**prefindex / 2 #1/pi**2/2 from FFTlog, 4 from Gauss Quad

        tosave[:, ichi1] = Cl


    np.savetxt('../output/cldeltachiphi/%d.txt'%index, tosave)

    if rank == 0: print('Time taken for index %d = '%index, time()-begin)


comm.barrier()

if rank ==0:
    cldeltaphi = np.zeros((t_.size, ell_.size, t_.size))
    for i in range(t_.size):
        cldeltaphi[i] = np.loadtxt('../output/cldeltachiphi/%d.txt'%i)

    cldeltaphi = np.swapaxes(cldeltaphi, 0, 1)

    np.save('../output/cm_clmesh/cldeltachiphi', cldeltaphi)


