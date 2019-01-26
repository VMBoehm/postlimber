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


#Kernels
def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) * (1.+z_chi(xi))


####################
outpath = '../output/clphipsi2/'
try: os.makedirs(outpath)
except: pass


chi1maxs = t_*chi_cmb
chi2s = t_*chi_cmb
chi1maxsplit = np.array_split(chi1maxs, wsize)
chiindex = np.arange(t_.size)
indexsplit = np.array_split(chiindex, wsize)
maxload = max(np.array([len(i) for i in indexsplit]))
if rank == 0: print('Maxload = ', maxload)
if rank == 0: np.savetxt(outpath + 'chis.txt', chi1maxs)
if rank == 0: print('chi1 split amongst ranks = ', chi1maxsplit)

nushift = 0
prefindex = 2

t1d = t_.reshape(-1, 1, 1)
w1d = w1.reshape(-1, 1, 1)
r2d = t_.reshape(1, -1, 1)
#chi2s = chi2s.reshape(1, -1, 1)
chi2s = chi_cmb

begin = time()

I0_lrtc = np.swapaxes(I0_ltrc, 1, 2)

for index in indexsplit[rank]:
    #print('Rank %d for index '%rank, index)
    chi1max = chi1maxs[index]
    
    chi1fac00 = lensing_kernel(t1d*r2d*chi_cmb, chi1max) *D_chi(t1d*r2d*chi_cmb)
    if rank == 0: print(chi1fac00.shape)
    chi1fac01 = lensing_kernel(1/t1d*r2d*chi_cmb, chi1max) * D_chi(1/t1d*r2d*chi_cmb)
    chi1fac01 = chi1fac01 * t1d**((nushift + nu_n_).reshape(1, 1, -1)-2)
    if rank == 0: print(chi1fac01.shape)
    chi1fac = chi1fac00 + chi1fac01
    if rank == 0: print(chi1fac.shape) 

    chi2fac = (r2d*chi_cmb)**(1-(nushift + nu_n_.reshape(1, 1, -1)))
    chi2fac *= D_chi(r2d*chi_cmb)
    chi2fac *= (1 + z_chi(r2d*chi_cmb))
    if rank == 0: print(chi2fac.shape) 
    
    chifac = w1d * chi1fac* chi2fac
    if rank == 0: print(chifac.shape) 
    if rank == 0: print('I0_ltrc ', I0_ltrc[1].shape) 

    Cl = np.zeros((ell_.size, t_.size))
    for ii in range(ell_.size):
        Cl[ii] = np.sum( chifac * I0_ltrc[ii], axis=(0, 2))
        #Cl[ii] = np.sum( chifac * I0_lrtc[ii], axis=(1, 2))

    Cl *= chi1max *1./np.pi**2/2.* prefac**prefindex / 2 #1/pi**2/2 from FFTlog, 4 from Gauss Quad                                                                                                                                                                                    
    np.savetxt(outpath + '%d.txt'%index, Cl, fmt='%0.4e', header='ell, chi2')

##
###Following also works, but its basically doing same as clphipsi.py
##r2d = 1
##
##I_ltc = np.squeeze(I0_ltrc)
##for index in indexsplit[rank]:
##    print('Rank %d for index '%rank, index)
##    chi1max = chi1maxs[index]
##    
##    Cl = np.zeros((ell_.size, t_.size))
##
##    begin = time()
##    for ichi, chi2s in enumerate(t_*chi_cmb):
##        chi1fac00 = lensing_kernel(t1d*r2d*chi2s, chi1max) *D_chi(t1d*r2d*chi2s)
##        if rank == 0: print(chi1fac00.shape)
##        chi1fac01 = lensing_kernel(1/t1d*r2d*chi2s, chi1max) * D_chi(1/t1d*r2d*chi2s)
##        chi1fac01 = chi1fac01 * t1d**((nushift + nu_n_).reshape(1, 1, -1)-2)
##        if rank == 0: print(chi1fac01.shape)
##        chi1fac = chi1fac00 + chi1fac01
##        if rank == 0: print(chi1fac.shape) 
##        
##        chi2fac = (r2d*chi2s)**(1-(nushift + nu_n_.reshape(1, 1, -1)))
##        chi2fac *= D_chi(r2d*chi2s)
##        chi2fac *= (1 + z_chi(r2d*chi2s))
##        if rank == 0: print(chi2fac.shape) 
##        
##        chifac = w1d * chi1fac* chi2fac
##        if rank == 0: print(chifac.shape) 
##        if rank == 0: print('I0_ltrc ', I0_ltrc[1].shape) 
##        
##        for ii in range(ell_.size):
##            #Cl[ii, ichi] = np.sum( chifac * I_ltc[ii])
##            Cl[ii, ichi] = np.sum( chifac[:, 0, :] * I0_ltrc[ii][:, 0, :])
##
##    Cl *= chi1max *1./np.pi**2/2.* prefac**prefindex / 2 #1/pi**2/2 from FFTlog, 4 from Gauss Quad                                                                                                                                                                                    
##    np.savetxt(outpath + '%d.txt'%index, Cl, fmt='%0.4e', header='ell, chi2')
##    if rank == 0: print('Time taken for index %d = '%index, time()-begin)
##
##
##
