#!/bin/bash
module load python/3.6-anaconda-4.4
paramfile=../settings/params_lsstall_cmblens_simple_bias_zmax10886.pkl
srun -n 25 python -u clphiphi_VB_parallel_source.py $paramfile
srun -n 100 python -u clphigal_VB_parallel.py $paramfile
srun -n 100 python -u clpsiphi_VB_parallel.py $paramfile
#paramfile=../settings/params_cross_gaussgal_z17_sigma2_deltalens_z17simple_bias_zmax25.pkl
#srun -n 25 python -u clphiphi_VB_parallel_source.py $paramfile
#srun -n 100 python -u clphigal_VB_parallel.py $paramfile
#srun -n 100 python -u clpsiphi_VB_parallel.py $paramfile
#paramfile=../settings/params_cross_gaussgal_z17_sigma2_deltalens_z23simple_bias_zmax25.pkl
#srun -n 25 python -u clphiphi_VB_parallel_source.py $paramfile
#srun -n 100 python -u clphigal_VB_parallel.py $paramfile
#srun -n 100 python -u clpsiphi_VB_parallel.py $paramfile
#paramfile=../settings/params_cross_gaussgal_z7_sigma2_deltalens_z13simple_bias_zmax15.pkl
#srun -n 25 python -u clphiphi_VB_parallel_source.py $paramfile
#srun -n 100 python -u clphigal_VB_parallel.py $paramfile
#srun -n 100 python -u clpsiphi_VB_parallel.py $paramfile


