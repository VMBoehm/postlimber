import numpy as np
from mpi4py import MPI
import os, sys
import json
from scipy.interpolate import interp1d
from scipy.integrate import simps, quadrature
import time

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


# ### Saving here

I0_ltc = np.squeeze(I0_ltrc)
nushift = 0 
t2d = t_.reshape(1, -1, 1)

chis = t_*chi_cmb
chips = t_*chi_cmb

Gsave = []

#Check if file exists. Otherwise create it

try: 
    print('saved file found')
    Gsave = np.load('../output/G-Ilnut_nd0.npy')
except Exception as e:
    print(e)
    print('Creating G matrix')
    begin = time.time()
    for ichim, chimax in enumerate(chis[:]):
        if ichim %10 == 0:
            print(ichim)

        chi1fac0 = lensing_kernel(chips, chimax)       #*D_chi(chi1s) #Move this inside the sum

        result = []
        for ii, chi in enumerate(chips):

            chifac = chi**(1-(nushift + nu_n_.reshape(1, 1, -1))) * D_chi(chi) #Move this c_n inside the loop

            chi2fac00 = (lensing_kernel(t2d*chi, chimax)*D_chi(t2d*chi))
            chi2fac01 = (lensing_kernel(1/t2d*chi, chimax)*D_chi(1/t2d*chi))* t2d**(nu_n_.reshape(1,  -1)-2)
            chi2fac0 = chi2fac00 + chi2fac01
            fac = w1.reshape(1, -1, 1)*chi2fac0*chifac
            integrand = fac * I0_ltc
            result.append(integrand.sum(axis = (1, 2)))

        Gsave.append(np.array(result).T)
        if ichim %10 == 0:
            print(time.time()-begin)

    Gsave = np.swapaxes(np.array(Gsave), 0, 1)

    np.save('../output/G-Ilnut_nd0', Gsave)



####################


#####
##Check To get Clkk

chi2 = chi_cmb*t_.reshape(1, 1, -1)
w2 = w1.reshape(1, 1, -1)
result = (w2*Gsave*lensing_kernel(chi2, chi_cmb)).sum(axis=2) *chi_cmb /2/np.pi**2
result *= prefac**2
clphiphi = result[:, -1]

#
ll = ell_
ell, clpp = np.loadtxt('../output/class_output.dat')[:2]
import matplotlib.pyplot as plt
plt.loglog(ell_,(ell_*(ell_+1))**2*clphiphi,ls='-', lw=1.5, label='Gee result')
plt.loglog(ell,(ell*(ell+1))**2*clpp,ls='--', lw=2, label='class')
plt.legend()
plt.loglog()
plt.xlim(1,2000)
plt.grid(which='both')
plt.savefig('../figures/clphiphi_Gsave.png')
