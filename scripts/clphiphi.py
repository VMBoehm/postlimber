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
outpath = './output/clphiphi2/'
try: os.makedirs(outpath)
except: pass

chi1maxs = t_*chi_cmb
chi2maxs = t_*chi_cmb
chi1maxsplit = np.array_split(chi1maxs, wsize)
chiindex = np.arange(t_.size)
indexsplit = np.array_split(chiindex, wsize)
maxload = max(np.array([len(i) for i in indexsplit]))
if rank == 0: print('Maxload = ', maxload)
if rank == 0: np.savetxt(outpath + 'chis.txt', chi1maxs)
if rank == 0: print('chi1 split amongst ranks = ', chi1maxsplit)

#indexsplit = [[99]]
kernel1 = lensing_kernel
kernel2 = lensing_kernel

for index in indexsplit[rank]:
    print('Rank %d for index '%rank, index)
    chi1max = chi1maxs[index]
    
    begin=time()
    tosave = np.zeros((ell_.size, chi2maxs.size+1))
    tosave[:, 0] = ell_
    for ichi2, chi2max in enumerate(chi2maxs[::]):
        
        Cl = getcl(kernel1, kernel2, chi1max, chi2max, nushift=0, prefindex=2)
        tosave[:, ichi2+1] = Cl

    np.savetxt(outpath + '%d.txt'%index, tosave, fmt='%0.4e', header='ell, chi2')

    if rank == 0: print('Time taken for index %d = '%index, time()-begin)

