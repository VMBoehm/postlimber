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

def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) * (1 + z_chi(xi))

wsize = 1
chi1maxs = t_*chi_cmb
chi2s = t_*chi_cmb
chi1maxsplit = np.array_split(chi1maxs, wsize)
chiindex = np.arange(t_.size)
indexsplit = np.array_split(chiindex, wsize)
maxload = max(np.array([len(i) for i in indexsplit]))

nushift = 0
prefindex = 2

t1d = t_.reshape(-1, 1, 1)
w1d = w1.reshape(-1, 1, 1)
r2d = t_.reshape(1, -1, 1)
chi2s = chi2s.reshape(1, -1, 1)

myresults = np.zeros((t_.size, ell_.size, t_.size))

begin = time()
chi2fac = (r2d*chi2s)**(1**(1-(nushift + nu_n_.reshape(1, -1))))
chi2fac *= D_chi(r2d*chi2s)
chi2fac *= (1 + z_chi(r2d*chi2s))

mfacs = w1d*chi2fac*I0_ltrc
chi1mesh = t1d*r2d*chi2s
ichi1mesh = 1/t1d*r2d*chi2s
Dchi1mesh =  D_chi(chi1mesh)
iDchi1mesh =  D_chi(ichi1mesh)
t1dnu = t1d**((nushift + nu_n_).reshape(1, 1, -1)-2)

for index in chiindex:
    print('%d for index '%index)
    chi1max = chi1maxs[index]
    result = np.zeros((ell_.size, chi2s.size))
    
    chi1fac00 = lensing_kernel(chi1mesh, chi1max) *Dchi1mesh
#     print(chi1fac00.shape)
    chi1fac01 = lensing_kernel(ichi1mesh, chi1max) * iDchi1mesh
#     print(chi1fac01.shape)
    chi1fac01 = chi1fac01 * t1dnu
#     print(chi1fac01.shape)
    chi1fac = chi1fac00 + chi1fac01
#     print(chi1fac.shape)                                                                                                                                                                                                                                                             

    matrix = mfacs*chi1fac
#     print(matrix.shape)                                                                                                                                                                                                                                                              
    Cl = np.sum(matrix, axis=(1, 3))
    Cl *= chi1max *1./np.pi**2/2.* prefac**prefindex / 4 #1/pi**2/2 from FFTlog, 4 from Gauss Quad                                                                                                                                                                                    
    myresults[:] = Cl

print('Time taken = ', time() - begin)
#     np.savetxt(outpath + '%d.txt'%index, result, fmt='%0.4e', header='ell, chi2')

myresults = np.swapaxes(myresults, 0, 1)

np.save(outpath + 'clphipsi2', myresults)
