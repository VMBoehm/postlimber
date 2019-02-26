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

####################


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


#galaxy_kernel = lambda xi, xmax : lsst_kernel_cb(xi)
if params.bias == 'simple': bias = simple_bias
elif params.bias == 'constant': bias = constant_bias
#kernel = dNdz_LSST(params.lsst)
#galaxy_kernel = lambda xi, xmax: kernel(xi)*bias(xi)
galaxy_kernel = lambda xi, xmax: gal_clus(dNdz_LSST, bias, params.lsst)(xi)
#galaxy_kernel = lambda xi, xmax: lsst_kernel_cb(xi) * simple_bias(xi)
#galaxy_kernel = lambda xi, xmax: lsst_kernel_cb(xi) * simple_bias(xi)

ofolder = params.ofolder
if rank == 0: print(ofolder)
try: os.makedirs(ofolder)
except: pass
##


####################


chi1max = chi_cmb
chi2max = chi_cmb

r1d, t1d = t_.reshape(1, -1), t_.reshape(-1, 1)
w11, w12 = w1.reshape(1, -1), w1.reshape(-1, 1)
# inflate by one dimensions (nu_n)
#r2d, t2d = np.expand_dims(r1d, 2), np.expand_dims(t1d, 2)
#w11, w12 = np.expand_dims(w11, 2), np.expand_dims(w12, 2)


indexes = np.arange(ell_.size)
ellsplit = np.array_split(ell_, wsize)
indexsplit = np.array_split(indexes, wsize)

cl22 = np.zeros((ell_.size, ell_.size))

chipkernel = galaxy_kernel(r1d*chi_cmb, chi_cmb)
chikernel = lensing_kernel(t1d*chi_cmb, chi_cmb)
fac1 = clpsipmesh
allfacs = fac1 * chikernel * chipkernel

print(chikernel.shape, chipkernel.shape, allfacs.shape)

for il in indexsplit[rank]:
    
    fac2 = clpdelmesh[il]
    matrix = w11*w12*fac2*allfacs
    Cl = matrix.sum(axis=(1, 2)) / 4
    cl22[:, il] = Cl #clphidel (L-l) is the second index here

cl22 *= chi_cmb**2

result = comm.gather(cl22, root=0)

if rank ==0:
    Cl22 = np.concatenate([result[ii][:, indexsplit[ii]] for ii in range(wsize)], axis=-1)
    print(Cl22.shape)
    np.save(ofolder + '/cl22B', Cl22)
