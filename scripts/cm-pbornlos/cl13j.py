#Eq. 4.5 of the text

import numpy as np
from mpi4py import MPI
import os, sys
import json
from scipy.interpolate import interp1d
from scipy.integrate import simps, quadrature
from time import time

#My imports
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
clpdelmesh *= -1 #missed this factor in clphidelta file, from \psi -> \phi
clppmesh = np.load('../../G_matrices/clphiphi_parallel.npy')
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

t1d = t_.reshape(-1, 1)
w1d = w1.reshape(-1, 1)


cl13j = np.zeros((ell_.size, ell_.size))

kernel = galaxy_kernel(t1d*chi_cmb, chi_cmb)
clppfac = np.expand_dims(clppmesh[:, :,  -1], 2)

if rank == 0:
    for il in range(ell_.size):

        fac = np.diag(clpdelmesh[il]).reshape(-1, 1)
        matrix = w1d*fac*kernel*clppfac
        print(matrix.shape)
        Cl = matrix.sum(axis=(1, 2)) / 2
        cl13j[:, il] = Cl     #\phidel (l) is the second index, \phiphi (L) is the first index

    cl13j *= chi_cmb

    cl13j *= -0.5 #Prefactor in front of the equation
    np.save(ofolder+'/cl13j', cl13j)
