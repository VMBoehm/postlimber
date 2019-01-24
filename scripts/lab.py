import numpy as np
import os, json
import tools
from scipy.interpolate import interp1d

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
