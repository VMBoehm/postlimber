#!/bin/bash
module load python/3.6-anaconda-4.4
paramfile=../settings/params_cross_gaussgal_z20_sigma5_cmblenssimple_bias_zmax10886.pkl
### these only need to be run for new chimax
srun -n 100 python -u clphiphi_VB_parallel_split.py $paramfile
srun -n 100 python -u clphiphi_VB_parallel_split2.py $paramfile
srun -n 100 python -u clphiphi_VB_parallel.py $paramfile
srun -n 100 python -u clpsiphi_VB_parallel.py $paramfile
srun -n 100 python -u clphidelta_VB_parallel_MB2.py $paramfile
paramfile=../settings/params_cross_gaussgal_z10_sigma1_deltalens_z7simple_bias_zmax15.pkl
### these only need to be run for new chimax
srun -n 100 python -u clphiphi_VB_parallel_split.py $paramfile
srun -n 100 python -u clphiphi_VB_parallel_split2.py $paramfile
srun -n 100 python -u clphiphi_VB_parallel.py $paramfile
srun -n 100 python -u clpsiphi_VB_parallel.py $paramfile
srun -n 100 python -u clphidelta_VB_parallel_MB2.py $paramfile
paramfile=../settings/params_cross_gaussgal_z20_sigma1_deltalens_z17simple_bias_zmax25.pkl
### these only need to be run for new chimax
srun -n 100 python -u clphiphi_VB_parallel_split.py $paramfile
srun -n 100 python -u clphiphi_VB_parallel_split2.py $paramfile
srun -n 100 python -u clphiphi_VB_parallel.py $paramfile
srun -n 100 python -u clpsiphi_VB_parallel.py $paramfile
srun -n 100 python -u clphidelta_VB_parallel_MB2.py $paramfile

