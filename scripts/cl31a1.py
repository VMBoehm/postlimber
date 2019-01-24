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

outpath = './output/'

#Kernels
def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) * (1.+z_chi(xi))

def Gauss_redshift(z0,sigma_z):
    def z_kernel(z):
        return 1./np.sqrt(2.*np.pi)/sigma_z*np.exp(-(z-z0)**2/2./sigma_z**2)
    return z_kernel

def setup_gal_kernel(z_kernel,ximax=5):
    def chi_kernel(xi, ximax):
        return z_kernel(z_chi(xi))*dz_dchi(xi)
    return chi_kernel

gmean, gwidth = 2.0, 1.0
gal_kernel = setup_gal_kernel(Gauss_redshift(gmean,gwidth))



####################


def setup_kernel2():
    def kernel2(xi, ximax):
        return 1
    return kernel2

kernel1 = gal_kernel
kernel2 = setup_kernel2()
chi1max = chi_cmb
chi2max = chi_cmb
nushift = 2
prefindex = 1

Cl = getcl(kernel1, kernel2, chi1max, chi2max, nushift, prefindex)

##
print(Cl)
