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
clppmesh = np.load('../output/clphiphi.npy')
chis = np.loadtxt('../output/chis.txt')
indexchi = {}
for i in range(chis.size): indexchi[chis[i]] = i


def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) * (1.+z_chi(xi))

##
galaxy_kernel = lambda xi, xmax : lsst_kernel_cb(xi)

def setup_galaxy_kernel(lindex):
    chifac = np.diag(clppmesh[int(lindex)]).reshape(-1, 1) #reshape is (-1, 1) if its for second kernel. (1, -1) for first kernel
    kernel = lambda  xi, xmax :  chifac * galaxy_kernel(xi, xmax)
    return kernel




####################


kernel1 = lensing_kernel
chi1max = chi_cmb
chi2max = chi_cmb
nushift = 2
prefindex = 1
In_ltrc = [I0_ltrc, None, I2_ltrc, None, I4_ltrc]
I_ltrc = In_ltrc[nushift]
Clmesh = []

for lindex in range(ell_.size):
    print(lindex)
    kernel2 = setup_galaxy_kernel(lindex)
    Cl = getcl(kernel1, kernel2, chi1max, chi2max, nushift, prefindex)
    Clmesh.append(Cl)
##
Clmesh = np.array(Clmesh).T #to save L as the first index

np.save('../output/cl13a', Clmesh)
