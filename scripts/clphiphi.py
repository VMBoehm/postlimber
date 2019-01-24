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

##r2d, t2d = np.meshgrid(t_,t_)
##w11, w12 = np.meshgrid(w1,w1)
### inflate by one dimensions (nu_n)
##r2d, t2d = np.expand_dims(r2d, 2), np.expand_dims(t2d, 2)
##w11, w12 = np.expand_dims(w11, 2), np.expand_dims(w12, 2)
###I0_ltrc  = np.swapaxes(I0_lcrt, 1, 3)
##

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
for index in indexsplit[rank]:
    print(index)
    chi1max = chi1maxs[index]
    
    begin=time()

    tosave = np.zeros((ell_.size, chi2maxs.size+1))
    tosave[:, 0] = ell_
    for ichi2, chi2max in enumerate(chi2maxs[::]):
        
##        chi1fac0 = (lensing_kernel(r2d*chi1max, chi1max)*D_chi(r2d*chi1max))
##        chi1fac0 = chi1fac0 *(r2d*chi1max)**(1-nu_n_.reshape(1, 1, -1))
##
##        chi2fac00 = (lensing_kernel(t2d*r2d*chi1max, chi2max)*D_chi(r2d*t2d*chi1max))
##        chi2fac01 = (lensing_kernel(1./t2d*r2d*chi1max, chi2max)*D_chi(r2d*1./t2d*chi1max))
##        chi2fac01 = chi2fac01 * t2d**(nu_n_.reshape(1, 1, -1)-2)
##        chi2fac0  = chi2fac00 + chi2fac01
##
##        chifacs = w11*w12*chi1fac0* chi2fac0
##
##        result=np.zeros_like(ell_)
##        for ii  in range(ell_.size):        
##            result[ii] = np.sum(chifacs*I0_ltrc[ii])
##
##        tosave[:, ichi2+1] = chi1max * result*1./np.pi**2/2. * prefac**2
##
        kernel1 = lensing_kernel
        kernel2 = lensing_kernel
        Cl = getcl(kernel1, kernel2, chi1max, chi2max, nushift=0, prefindex=2)
        tosave[:, ichi2+1] = Cl

    np.savetxt(outpath + '%d.txt'%index, tosave, fmt='%0.4e', header='ell, chi2')

    if rank == 0: print('Time taken for index %d = '%index, time()-begin)

