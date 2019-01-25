import numpy as np
from lab import *


##

r2d, t2d = np.meshgrid(t_,t_)
w11, w12 = np.meshgrid(w1,w1)
# inflate by one dimensions (nu_n)
r2d, t2d = np.expand_dims(r2d, 2), np.expand_dims(t2d, 2)
w11, w12 = np.expand_dims(w11, 2), np.expand_dims(w12, 2)
#I0_ltrc  = np.swapaxes(I0_lcrt, 1, 3)

In_ltrc = [I0_ltrc, None, I2_ltrc, None, I4_ltrc]



def getcl(kernel1, kernel2, chi1max, chi2max, nushift, prefindex):

    I_ltrc = In_ltrc[nushift]

    chi1fac0 = (kernel1(r2d*chi1max, chi1max) * D_chi(r2d*chi1max))
    chi1fac0 = chi1fac0 * (r2d*chi1max)**(1-(nushift + nu_n_.reshape(1, 1, -1)))

    chi2fac00 = (kernel2(t2d*r2d*chi1max, chi2max) * D_chi(r2d*t2d*chi1max))
    chi2fac01 = (kernel2(1./t2d*r2d*chi1max, chi2max) * D_chi(r2d/t2d*chi1max))
    chi2fac01 = chi2fac01 * t2d**((nushift + nu_n_).reshape(1, 1, -1)-2)
    chi2fac0  = chi2fac00 + chi2fac01


    chifacs = w11*w12*chi1fac0* chi2fac0

    result=np.zeros_like(ell_)
    for ii  in range(ell_.size):        
        result[ii] = np.sum(chifacs*I_ltrc[ii])

    Cl = chi1max * result *1./np.pi**2/2.* prefac**prefindex / 4 #1/pi**2/2 from FFTlog, 4 from Gauss Quad
    
    return Cl
