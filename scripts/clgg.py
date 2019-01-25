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
#import kernels

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

galaxy_kernel = lambda xi, xmax : lsst_kernel_cb(xi)
#galaxy_kernel = lambda xi, xmax : gal_kernel(Gauss_redshift(2, 1.0))(xi)

####################



kernel1 = galaxy_kernel
kernel2 = galaxy_kernel
chi1max = chi_cmb
chi2max = chi_cmb
nushift = 4
prefindex = 0

Cl = getcl(kernel1, kernel2, chi1max, chi2max, nushift, prefindex)
##
print(Cl)


#
import matplotlib.pyplot as plt
plt.loglog(ell_,Cl,ls='-', lw=1.5, label='LSST Clgg')
plt.legend()
plt.loglog()
plt.xlim(1,2000)
plt.grid(which='both')
plt.savefig('../figures/clgg_lsst.png')
