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
outpath = '../output/clpsipsi/'
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
I_ltc = np.squeeze(I0_ltrc)

print(I_ltc.shape)

t1d = t_.reshape(-1, 1)
w1d = w1.reshape(-1, 1)

Clmatrix = np.zeros((len(t_),len(ell_),len(t_)))

for index in indexsplit[rank]:
    print('Rank %d for index '%rank, index)
    chi1 = chi1maxs[index]
    
    Cl = np.zeros((ell_.size, t_.size))

    begin = time()
    chi1fac = (D_chi(chi1))
    chi1fac = chi1fac *chi1**(-(nushift + nu_n_.reshape(1, -1)))
    chi1fac *= (1 + z_chi(chi1))
    print(chi1fac.shape)

    chi2 = t1d*chi1
    chi2fac = D_chi(chi2)
    chi2fac = chi2fac * (1 + z_chi(chi2))
    print(chi2fac.shape)
    
    chifac = chi1fac * chi2fac
    print(chifac.shape)

    for ii in range(ell_.size):
        matrix =  chifac * I_ltc[ii]
        Cl[ii] = np.sum(matrix, axis=-1)
        
    Cl *= 1./np.pi**2/2.* prefac**prefindex 
    
    Clmatrix[index] = Cl
    np.savetxt(outpath + '%d.txt'%index, Cl, fmt='%0.4e', header='ell, chi2')

    if rank == 0: print('Time taken for index %d = '%index, time()-begin)



comm.barrier()
if rank ==0:
    cl = []
    for i in range(100):
        cl.append(np.loadtxt('../output/clpsipsi/%d.txt'%i))
    cl = np.swapaxes(np.array(cl), 0, 1)
    np.save('../output/clpsipsi.npy', cl)
