#Eq. 4.5 of the text

import numpy as np
from mpi4py import MPI
import os, sys
import json
from scipy.interpolate import interp1d
from scipy.integrate import simps, quadrature
from time import time

#My imports
import sys
sys.path.append('../')
from lab import *
import tools
from getcl import getcl
import kernels
import params

#MPI
comm = MPI.COMM_WORLD
rank, wsize = comm.rank, comm.size
print(wsize, rank)

#Kernels
clpdelmesh = np.load('../../G_matrices/clphidelta.npy')
clpdelmesh *= -1 #Missed -1 in clphidelta that comes from \psi->\phi 
clppsimesh = np.load('../../G_matrices/clphipsi.npy')
clppsimesh *= -1 #Missed -1 in clphipsi that comes from \psi->\phi                                                                      
clpsipmesh = np.swapaxes(clppsimesh, 1, 2)
chis = np.loadtxt('../../output/chis.txt')


def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) #* (1.+z_chi(xi)) This is already in the files

##
#galaxy_kernel = lambda xi, xmax : lsst_kernel_cb(xi)
if params.bias == 'simple': bias = simple_bias
elif params.bias == 'constant': bias = constant_bias
galaxy_kernel = lambda xi, xmax: gal_clus(dNdz_LSST, bias, params.lsst)(xi)

ofolder = params.ofolder
if rank == 0: print(ofolder)
try: os.makedirs(ofolder)
except: pass
##


####################


chi1max = chi_cmb
chi2max = chi_cmb

r2d, t2d = t_.reshape(1, -1), t_.reshape(-1, 1)
w11, w12 = w1.reshape(1, -1), w1.reshape(-1, 1)


indexes = np.arange(ell_.size)
ellsplit = np.array_split(ell_, wsize)
indexsplit = np.array_split(indexes, wsize)

cl31ab = np.zeros((ell_.size, ell_.size))

chipkernel = galaxy_kernel(r2d*chi_cmb, chi_cmb)
chikernel = lensing_kernel(t2d*chi_cmb, chi_cmb)
fac1 = clpdelmesh
allfacs = fac1 * chikernel * chipkernel
print(chikernel.shape, chipkernel.shape, allfacs.shape)

for il in indexsplit[rank]:
    
    fac2 = np.diag(clppsimesh[il]).reshape(-1, 1)
    matrix = w11*w12*fac2*allfacs
    Cl = matrix.sum(axis=(1, 2)) / 4
    cl31ab[:, il] = Cl

cl31ab *= chi_cmb**2

result = comm.gather(cl31ab, root=0)

if rank ==0:
    Cl31ab = np.concatenate([result[ii][:, indexsplit[ii]] for ii in range(wsize)], axis=-1)
    print(Cl31ab.shape)
    np.save(ofolder + '/cl31aB', Cl31ab)
