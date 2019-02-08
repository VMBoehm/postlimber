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
clpdelmesh = np.load('../G_matrices/clphidelta.npy')
clppmesh = np.load('../G_matrices/clphiphi_parallel.npy')
chis = np.loadtxt('../output/chis.txt')


def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) #* (1.+z_chi(xi)) This is already in the files

##
galaxy_kernel = lambda xi, xmax : lsst_kernel_cb(xi)


####################


chi1max = chi_cmb
chi2max = chi_cmb

t1d = t_.reshape(-1, 1)
w1d = w1.reshape(-1, 1)


cl13j = np.zeros((ell_.size, ell_.size))

kernel = galaxy_kernel(t1d*chi_cmb, chi_cmb)
clppfac = np.expand_dims(clppmesh[:, :,  -1], 2)

for il in range(ell_.size):
    
    fac = np.diag(clpdelmesh[il]).reshape(-1, 1)
    matrix = w1d*fac*kernel*clppfac
    print(matrix.shape)
    Cl = matrix.sum(axis=(1, 2)) / 2
    cl13j[:, il] = Cl

cl13j *= chi_cmb

np.save('../output/cm_clmesh/cl13j', cl13j)
