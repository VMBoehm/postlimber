#Eq. 4.3 of the text

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
clppmesh = np.load('../G_matrices/clphiphi.npy')
chis = np.loadtxt('../output/chis.txt')
indexchi = {}
for i in range(chis.size): indexchi[chis[i]] = i


def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) * (1.+z_chi(xi))

##
galaxy_kernel = lambda xi, xmax : lsst_kernel_cb(xi)





####################


kernel1 = lensing_kernel
kernel2 = galaxy_kernel
chi1max = chi_cmb
chi2max = chi_cmb
nushift = 2
prefindex = 1
In_ltrc = [I0_ltrc, None, I2_ltrc, None, I4_ltrc]
I_ltrc = In_ltrc[nushift]
print(I_ltrc.shape)

#r2d, t2d = np.meshgrid(t_,t_)
#w11, w12 = np.meshgrid(w1,w1)
r2d, t2d = t_.reshape(1, -1), t_.reshape(-1, 1)
w11, w12 = w1.reshape(1, -1), w1.reshape(-1, 1)
# inflate by one dimensions (nu_n)
r2d, t2d = np.expand_dims(r2d, 2), np.expand_dims(t2d, 2)
w11, w12 = np.expand_dims(w11, 2), np.expand_dims(w12, 2)


indexes = np.arange(ell_.size)
ellsplit = np.array_split(ell_, wsize)
indexsplit = np.array_split(indexes, wsize)

cl31aA = np.zeros((ell_.size, ell_.size))

for il in indexsplit[rank]:
    if rank ==0:     print('Rank %d for index '%rank, il, ' of ', indexsplit[rank])
    chi1fac0 = (kernel1(r2d*chi1max, chi1max) * D_chi(r2d*chi1max))
    chi1fac0 = chi1fac0 * (r2d*chi1max)**(1-(nushift + nu_n_.reshape(1, 1, -1)))

    chi2fac00 = (kernel2(t2d*r2d*chi1max, chi2max) * D_chi(r2d*t2d*chi1max))
    chi2fac01 = (kernel2(1./t2d*r2d*chi1max, chi2max) * D_chi(r2d/t2d*chi1max))
    chi2fac01 = chi2fac01 * t2d**((nushift + nu_n_).reshape(1, 1, -1)-2)
    chi2fac0  = chi2fac00 + chi2fac01


    chifacs = w11*w12*chi1fac0* chi2fac0
    phifacs = np.diag(clppmesh[il]).reshape(1, -1)
    phifacs = np.expand_dims(phifacs, -1)
    #print(chifacs.shape, phifacs.shape)
    allfacs = chifacs*phifacs
    result=np.zeros_like(ell_)
    for ii  in range(ell_.size):        
        result[ii] = np.sum(allfacs*I_ltrc[ii])

    Cl = chi1max * result *1./np.pi**2/2.* prefac**prefindex / 4 #1/pi**2/2 from FFTlog, 4 from Gauss Quad
    cl31aA[:, il] = Cl


result = comm.gather(cl31aA, root=0)

if rank ==0:
    Cl31aA = np.concatenate([result[ii][:, indexsplit[ii]] for ii in range(wsize)], axis=-1)
    print(Cl31aA.shape)

    np.save('../output/cl31aA-v2', Cl31aA)
