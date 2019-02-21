##Save the files for the Cl_phi_delta(r2d*t2d*chi_cmb, chi_cmb)


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
import params

#MPI
comm = MPI.COMM_WORLD
rank, wsize = comm.rank, comm.size
print(wsize, rank)


#Kernels
def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) * (1.+z_chi(xi))

#galaxy_kernel = lambda xi, xmax : lsst_kernel_cb(xi)
if params.bias == 'simple': bias = simple_bias
elif params.bias == 'constant': bias = constant_bias
galaxy_kernel = lambda xi, xmax: gal_clus(dNdz_LSST, bias, params.lsst)(xi)

outpath = '../../output/clphideltacmb/'
ofolder = '../../output/cm_clmesh/'
try: os.makedirs(outpath)
except: pass
try: os.makedirs(ofolder)
except: pass


In_ltrc = [I0_ltrc, None, I2_ltrc, None, I4_ltrc]

r2d, t2d = t_.reshape(1, -1), t_.reshape(-1, 1)
w11, w12 = w1.reshape(1, -1), w1.reshape(-1, 1)
# inflate by one dimensions (nu_n)
r2d, t2d = np.expand_dims(r2d, 2), np.expand_dims(t2d, 2)
w11, w12 = np.expand_dims(w11, 2), np.expand_dims(w12, 2)

####################


#t_ = t_[:10]

chi1maxs = t_*chi_cmb
chi1maxsplit = np.array_split(chi1maxs, wsize)
chiindex = np.arange(t_.size)
indexsplit = np.array_split(chiindex, wsize)

maxload = max(np.array([len(i) for i in indexsplit]))
if rank == 0: print('Maxload = ', maxload)
if rank == 0: print('chi1 split amongst ranks = ', chi1maxsplit)

nushift = 2
prefindex = 1

I_ltrc = In_ltrc[nushift]

kernel1 = lensing_kernel
kernel2 = lsst_kernel_cb 
#chi2max = chi_cmb

clphidel = np.zeros((ell_.size, t_.size, t_.size))
##
##for index in indexsplit[rank]:
##    print('Rank %d for index '%rank, index)
##    chi1max = chi1maxs[index]
##    
##    begin=time()
##
##    tosave = np.zeros((ell_.size, t_.size))
##    for ichi1, chi1maxt in enumerate(t_*chi1max):
##        if rank == 0: print(ichi1, chi1maxt)
##
##
##        chi1fac0 = (kernel1(r2d*chi1maxt, chi1maxt) * D_chi(r2d*chi1maxt))
##        chi1fac0 = chi1fac0 * (r2d*chi1maxt)**(1-(nushift + nu_n_.reshape(1, 1, -1)))
##
##        chi2fac00 = (kernel2(t2d*r2d*chi1maxt) * D_chi(r2d*t2d*chi1maxt))
##        chi2fac01 = (kernel2(1./t2d*r2d*chi1maxt) * D_chi(r2d/t2d*chi1maxt))
##        chi2fac01 = chi2fac01 * t2d**((nushift + nu_n_).reshape(1, 1, -1)-2)
##        chi2fac0  = chi2fac00 + chi2fac01
##
##
##        chifacs = w11*w12*chi1fac0* chi2fac0
##
##        result=np.zeros_like(ell_)
##        for ii  in range(ell_.size):        
##            result[ii] = (np.sum(chifacs*I_ltrc[ii]))
##
##        Cl = 2 * chi1maxt * result *1./np.pi**2/2.* prefac**prefindex / 4 #1/pi**2/2 from FFTlog, 4 from Gauss Quad
##
##        #np.savetxt(outpath + '/%d-%d.txt'%(index, ichi1), Cl)
##        tosave[:, ichi1] = Cl
##        clphidel[:, index, ichi1] = Cl
##
##
##    np.savetxt(outpath + '/%d.txt'%index, tosave, fmt='%0.4e', header='ell, chi2')
##
##
##    print('Time taken for index %d in rank %d= '%(index, rank), time()-begin)
##


comm.barrier()

if rank ==0:
    Clphidel = np.zeros((t_.size, ell_.size, t_.size))
    for i in range(t_.size):
        Clphidel[i] = np.loadtxt(outpath + '/%d.txt'%i)

    Clphidel = np.swapaxes(Clphidel, 0, 1)
    np.save(ofolder + '/clphideltacmb%s'%params.filext, Clphidel)


