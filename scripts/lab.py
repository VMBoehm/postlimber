import numpy as np
import os, json
import tools
from scipy.interpolate import interp1d

from kernelsnew import *

package_path = os.path.dirname(os.path.abspath(__file__))+'/'
dpath = package_path + '../PostBornEma/'

ell_, t_, nu_n_, I0_ltrc, I2_ltrc, I4_ltrc = tools.loadfftlogdata()
t1, w1 = tools.loadggwts()
assert(np.allclose(t_,t1))

#Setup cosmology dicts
with open(package_path + 'class_cosmo_b.json', 'r') as fp: cosmo_b = json.load(fp)
for key in cosmo_b: cosmo_b[key] = np.array(cosmo_b[key])
with open(package_path + 'cosmo_dict.json', 'r') as fp: cosmo_dict = json.load(fp)
for key in cosmo_dict: locals()[key] = cosmo_dict[key]



#prefactor for Cl_kk computation from Cl_dd
c      = 299792458./1000. # km/s
Omega_b   = omega_b/h**2
Omega_cdm = omega_cdm/h**2
Omega_m   = Omega_b+Omega_cdm
prefac = 1.5*Omega_m*(100.)**2/c**2 #without h


#Setup interpolating functions
class_z               = cosmo_b['z'][::-1]
class_chi             = cosmo_b['comov. dist.'][::-1]
class_D               = cosmo_b['gr.fac. D'][::-1]
class_H               = cosmo_b['H [1/Mpc]'][::-1]/h #already divided by c

chi_z = interp1d(class_z,class_chi*h,fill_value=0, bounds_error=False)
z_chi = interp1d(class_chi*h,class_z,fill_value=0, bounds_error=False)  # Mpc/h
D_chi = interp1d(class_chi*h,class_D,fill_value=0, bounds_error=False)    # growth
D_z   = interp1d(class_z,class_D,fill_value=0, bounds_error=False)

#
chi_cmb = chi_z(z_cmb)
dchi_dz=(class_chi[1::]-class_chi[0:-1])/(class_z[1::]-class_z[0:-1])*h
z_mean = (class_z[1::]+class_z[0:-1])/2
dz_dchi = interp1d(class_chi*h,class_H,fill_value=0, bounds_error=False)
dchi_dz = interp1d(z_mean,dchi_dz,fill_value=0, bounds_error=False)



##Kernels
def Gauss_redshift(z0,sigma_z):
    def z_kernel(z):
        return 1./np.sqrt(2.*np.pi)/sigma_z*np.exp(-(z-z0)**2/2./sigma_z**2)
    return z_kernel

def gal_kernel(z_kernel,ximax=5):
    def chi_kernel(xi):
        return z_kernel(z_chi(xi))*dz_dchi(xi)
    return chi_kernel


# 2) prospective LSST kernels
def dNdz_LSST(bin_num,dn_filename = '../LSSTdndzs/dndz_LSST_i27_SN5_3y'):
    if bin_num is "all":
        zbin, nbin = np.load(dn_filename+'tot_extrapolated.npy',encoding='latin1')
        norm                = np.trapz(nbin,zbin)
        mbin                = 'None'
    else:
        bins,big_grid,res   = np.load(dn_filename+'_extrapolated.npy',encoding='latin1')
        mbin                = bins[bin_num]
        zbin                = big_grid
        nbin                = res[bin_num]
        norm                = np.trapz(nbin,zbin)
    dndz                = interp1d(zbin, nbin/norm, kind='linear',bounds_error=False,fill_value=0.)
    #print('using z-bin', mbin, 'norm', norm)
    return dndz



def gal_clus(dNdz,b,bin_num):
    """
    dNdz: function returning function dndz for gicen bin number 
    b: function returning bias as function of z 
    bin_num: bin_number (either 'all' or 0-5)
    """
    p_z=dNdz(bin_num)
    def kernel(x):
        z = z_chi(x)
        return b(z)*p_z(z)*dz_dchi(z)

    return kernel

def simple_bias(z):
    return (1.+z)

def constant_bias(z,b=1.):
    return b



lsst_kernel_cb = gal_clus(dNdz_LSST, constant_bias, 'all')
