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
    return (xmax - xi)/(xmax*xi) * (xmax > xi) * (1.+z_chi(xi))


####################
outpath = '../output/clphipsi/'
try: os.makedirs(outpath)
except: pass


chi1maxs = t_*chi_cmb
chi2s = t_*chi_cmb
chi1maxsplit = np.array_split(chi1maxs, wsize)
chiindex = np.arange(t_.size)
indexsplit = np.array_split(chiindex, wsize)
maxload = max(np.array([len(i) for i in indexsplit]))
if rank == 0: print('Maxload = ', maxload)
if rank == 0: np.savetxt(outpath + 'chis.txt', chi1maxs)
if rank == 0: print('chi1 split amongst ranks = ', chi1maxsplit)


nushift = 0 
prefindex = 2

t1d = t_.reshape(-1, 1)
w1d = w1.reshape(-1, 1)
I_ltc = np.squeeze(I0_ltrc)
print(I_ltc.shape)

for index in indexsplit[rank]:
    print('Rank %d for index '%rank, index)
    chi1max = chi1maxs[index]
    
    result = np.zeros((ell_.size, chi2s.size))
    begin = time()
    for ichi2, chi2 in enumerate(chi2s):
        
        chi1fac00 = (lensing_kernel(t1d*chi2, chi1max) * D_chi(t1d*chi2))
        chi1fac01 = (lensing_kernel(1/t1d*chi2, chi1max) * D_chi(1/t1d*chi2))
        chi1fac01 = chi1fac01 * t1d**((nushift + nu_n_).reshape(1, -1)-2)
        chi1fac = chi1fac00 + chi1fac01
        #if rank == 0: print(chi1fac.shape)

        chi2fac = chi2**(1-(nushift + nu_n_.reshape(1, -1)))
        chi2fac *= D_chi(chi2)
        chi2fac *= (1 + z_chi(chi2))
        #if rank == 0: print(chi2fac.shape)
        
        
        Cl = np.zeros_like(ell_)
        for ii in range(ell_.size):
            matrix = w1d * chi2fac* chi1fac * I_ltc[ii]
            Cl[ii] = np.sum(matrix)
        
        #print(matrix.shape)
        Cl *= chi1max *1./np.pi**2/2.* prefac**prefindex /2 #1/pi**2/2 from FFTlog, 4 from Gauss Quad
        #Cl *= 1./np.pi**2/2.* prefac**prefindex /4 #1/pi**2/2 from FFTlog, 4 from Gauss Quad
        result[:, ichi2] = Cl

    np.savetxt(outpath + '%d.txt'%index, result, fmt='%0.4e', header='ell, chi2')

    if rank == 0: print('Time taken for index %d = '%index, time()-begin)


