#Eq. 4.5 of the text

import numpy as np
from mpi4py import MPI
import os, sys
import json
from scipy.interpolate import interp1d
from scipy.integrate import simps, quadrature
from scipy.interpolate import RectBivariateSpline as rbspline
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
clpsiphimesh = np.load('../../output/cm_clmesh/clpsiphicmb.npy')
cldelphimesh = np.load('../../output/cm_clmesh/cldeltachiphi.npy')
clphidelmesh = np.swapaxes(cldelphimesh, 1, 2)
chis = np.loadtxt('../../output/chis.txt')


def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) * (1.+z_chi(xi))

##
#galaxy_kernel = lambda xi, xmax : lsst_kernel_cb(xi)
#galaxy_kernel = lambda xi, xmax : lsst_kernel_cbn[0](xi)
if params.bias == 'simple': bias = simple_bias
elif params.bias == 'constant': bias = constant_bias
galaxy_kernel = lambda xi, xmax: gal_clus(dNdz_LSST, bias, params.lsst)(xi)

ofolder = params.ofolder
if rank == 0: print(ofolder)
try: os.makedirs(ofolder)
except: pass


In_ltrc = [I0_ltrc, None, I2_ltrc, None, I4_ltrc]

r1d, t1d = t_.reshape(1, -1), t_.reshape(-1, 1)
w11, w12 = w1.reshape(1, -1), w1.reshape(-1, 1)
grid = r1d*t1d*chi_cmb
# inflate by one dimensions (nu_n)
#r2d, t2d = np.expand_dims(r1d, 2), np.expand_dims(t1d, 2)
#w11, w12 = np.expand_dims(w11, 2), np.expand_dims(w12, 2)


indexes = np.arange(ell_.size)
ellsplit = np.array_split(ell_, wsize)
indexsplit = np.array_split(indexes, wsize)
####################


#chikernel = lensing_kernel
chipkernel = galaxy_kernel
chippkernel = lensing_kernel

chi1max = chi_cmb
chi2max = chi_cmb


chipkernel = chipkernel(r1d*chi_cmb, chi_cmb)
chipkernel *= (r1d * chi_cmb) #add factor of gauss integral
chippkernel = chippkernel(r1d*t1d*chi_cmb, r1d*chi_cmb) #Note the max chi 
if rank == 0: print(chipkernel.shape, chippkernel.shape, clpsiphimesh.shape)

allfacs =  chipkernel * chippkernel * clpsiphimesh
if rank == 0: print(allfacs.shape)

cl13 = np.zeros((ell_.size, ell_.size))
for il in indexsplit[rank]:

    if rank == 0: print('Rank %d for index '%rank, il, ' of ', indexsplit[rank])

    fac2 = clphidelmesh[il]
    matrix = w11*w12*fac2*allfacs
    Cl = matrix.sum(axis=(1, 2)) / 4
    cl13[:, il] = Cl #clphidel is the second index i.e. l is second, L is first index


cl13 *= chi_cmb

result = comm.gather(cl13, root=0)

if rank ==0:
    Cl13 = np.concatenate([result[ii][:, indexsplit[ii]] for ii in range(wsize)], axis=-1)
    print(Cl13.shape)
    np.save(ofolder + '/cl13b', Cl13)

