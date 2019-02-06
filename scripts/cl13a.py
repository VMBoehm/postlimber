#Eq. 4.5 of the text

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
import kernels

#MPI
comm = MPI.COMM_WORLD
rank, wsize = comm.rank, comm.size
print(wsize, rank)

outpath = './output/'

#Kernels
clppmesh = np.load('../G_matrices/clphiphi_parallel.npy')
chis = np.loadtxt('../output/chis.txt')
indexchi = {}
for i in range(chis.size): indexchi[chis[i]] = i


def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) * (1.+z_chi(xi))

##
galaxy_kernel = lambda xi, xmax : lsst_kernel_cb(xi)

def setup_galaxy_kernel(lindex):
    chifac = np.diag(clppmesh[int(lindex)]).reshape(-1, 1) #always modify first kernel with this reshape because broadcasting matches things with trailing index
    kernel = lambda  xi, xmax :  chifac * galaxy_kernel(xi, xmax)
    return kernel




####################


kernel2 = lensing_kernel
chi1max = chi_cmb
chi2max = chi_cmb
nushift = 2
prefindex = 1
In_ltrc = [I0_ltrc, None, I2_ltrc, None, I4_ltrc]
I_ltrc = In_ltrc[nushift]

indexes = np.arange(ell_.size)
ellsplit = np.array_split(ell_, wsize)
indexsplit = np.array_split(indexes, wsize)

cl13a = np.zeros((ell_.size, ell_.size))

for il in indexsplit[rank]:
    print('Rank %d for index '%rank, il, ' of ', indexsplit[rank])
    kernel1 = setup_galaxy_kernel(il)
    Cl = getcl(kernel1, kernel2, chi1max, chi2max, nushift, prefindex)

    cl13a[:, il] = Cl

##

result = comm.gather(cl13a, root=0)

if rank ==0:
    Cl13a = np.concatenate([result[ii][:, indexsplit[ii]] for ii in range(wsize)], axis=-1)
    print(Cl13a.shape)

    np.save('../output/cm_clmesh/cl13a', Cl13a)
