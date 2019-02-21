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
import sys
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
clppmesh = np.load('../../G_matrices/clphiphi_parallel.npy')
chis = np.loadtxt('../../output/chis.txt')
indexchi = {}
for i in range(chis.size): indexchi[chis[i]] = i


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
##



####################


kernel1 = lensing_kernel
kernel2 = galaxy_kernel
chi1max = chi_cmb
chi2max = chi_cmb
nushift = 2
prefindex = 1
In_ltrc = [I0_ltrc, None, I2_ltrc, None, I4_ltrc]
I_ltrc = In_ltrc[nushift]
Clmesh = []
if rank == 0 : print(I_ltrc.shape)

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

cl22 = np.zeros((ell_.size, ell_.size))

for il in indexsplit[rank]:

    iclpp = rbspline(t_*chi_cmb, t_*chi_cmb, clppmesh[il])
    #
    if rank == 0: print('Rank %d for index '%rank, il, ' of ', indexsplit[rank])
    chi1fac0 = (kernel1(r2d*chi1max, chi1max) * D_chi(r2d*chi1max))
    chi1fac0 = chi1fac0 * (r2d*chi1max)**(1-(nushift + nu_n_.reshape(1, 1, -1)))

    chi2fac00 = (kernel2(t2d*r2d*chi1max, chi2max) * D_chi(r2d*t2d*chi1max))
    phifac00 = np.stack([iclpp(i, i*t_, grid=False) for i in t_*chi_cmb], axis=1)
    chi2fac00 *= np.expand_dims(phifac00, 2)

    chi2fac01 = (kernel2(1./t2d*r2d*chi1max, chi2max) * D_chi(r2d/t2d*chi1max))
    phifac01 = np.stack([iclpp(i, i/t_, grid=False) for i in t_*chi_cmb], axis=1)
    chi2fac01 *= np.expand_dims(phifac01, axis=2)
    chi2fac01 = chi2fac01 * t2d**((nushift + nu_n_).reshape(1, 1, -1)-2)

    chi2fac0  = chi2fac00 + chi2fac01

    chifacs = w11*w12*chi1fac0* chi2fac0

    result=np.zeros_like(ell_)
    for ii  in range(ell_.size):        
        result[ii] = np.sum(chifacs*I_ltrc[ii])

    Cl = chi1max * result *1./np.pi**2/2.* prefac**prefindex / 4 #1/pi**2/2 from FFTlog, 4 from Gauss Quad
    cl22[:, il] = Cl


result = comm.gather(cl22, root=0)

if rank ==0:
    Cl22 = np.concatenate([result[ii][:, indexsplit[ii]] for ii in range(wsize)], axis=-1)
    print(Cl22.shape)
    np.save(ofolder + '/cl22A', Cl22)
