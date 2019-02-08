#Eq. 4.6, 4.7 of the text

import numpy as np
from mpi4py import MPI
import os, sys
import json
from scipy.interpolate import interp1d
from scipy.integrate import simps, quadrature
from scipy.interpolate import RectBivariateSpline as rbspline
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
chis = np.loadtxt('../output/chis.txt')


def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) * (1.+z_chi(xi))

##
galaxy_kernel = lambda xi, xmax : lsst_kernel_cb(xi)
#galaxy_kernel = lambda xi, xmax : lsst_kernel_cbn[0](xi)



####################


chikernel = lensing_kernel
chipkernel = galaxy_kernel
chippkernel = lensing_kernel
chipfacs = (chi_cmb*w1*chipkernel(t_*chi_cmb, chi_cmb)).reshape(1, 1, -1)/2.

chi1max = chi_cmb
chi2max = chi_cmb

nushift = 0
prefindex = 2
In_ltrc = [I0_ltrc, None, I2_ltrc, None, I4_ltrc]
I_ltrc = In_ltrc[nushift]
Clmesh = []
if rank == 0 : print(I_ltrc.shape)

r1d, t1d = t_.reshape(1, -1), t_.reshape(-1, 1)
w11, w12 = w1.reshape(1, -1), w1.reshape(-1, 1)
grid = r1d*t1d*chi_cmb
# inflate by one dimensions (nu_n)
r2d, t2d = np.expand_dims(r1d, 2), np.expand_dims(t1d, 2)
w11, w12 = np.expand_dims(w11, 2), np.expand_dims(w12, 2)


indexes = np.arange(ell_.size)
ellsplit = np.array_split(ell_, wsize)
indexsplit = np.array_split(indexes, wsize)

cl31 = np.zeros((ell_.size, ell_.size))
for il in indexsplit[rank]:

    if rank == 0: print('Rank %d for index '%rank, il, ' of ', indexsplit[rank])

    #Do the chi' integral first and get a 2D mesh of chi, chi'' corresponding to r2d*t2d
    iclpdelta = rbspline(t_*chi_cmb, t_*chi_cmb, clpdelmesh[il])    
    phidelfac = np.stack([iclpdelta(t_*i, t_*chi_cmb, grid=True) for i in t_*chi_cmb], axis=1)
    phidelfac *= chipfacs
    phidelfac = np.sum(phidelfac, axis=-1)
    
    chi1fac0 = (chikernel(r2d*chi1max, chi1max) * D_chi(r2d*chi1max))
    chi1fac0 = chi1fac0 * (r2d*chi1max)**(1-(nushift + nu_n_.reshape(1, 1, -1)))

    chi2fac00 = (chippkernel(t2d*r2d*chi1max, r2d*chi1max) * D_chi(r2d*t2d*chi1max)) ##Note the change in chimax of kernel
    chi2fac00 *= np.expand_dims(phidelfac, 2)

    #chi2fac0  = chi2fac00 + chi2fac01

    chifacs = w11*w12*chi1fac0* chi2fac00

    result=np.zeros_like(ell_)
    for ii  in range(ell_.size):        
        result[ii] = np.sum(chifacs*I_ltrc[ii])

    Cl = chi1max * result *1./np.pi**2/2.* prefac**prefindex / 4 ###1/pi**2/2 from FFTlog, 4 from Gauss Quad
    cl31[il, :] = Cl


result = comm.gather(cl31, root=0)

if rank ==0:
    Cl31 = np.concatenate([result[ii][:, indexsplit[ii]] for ii in range(wsize)], axis=-1)
    print(Cl31.shape)
    np.save('../output/cm_clmesh/cl31b-phiinterp', Cl31)