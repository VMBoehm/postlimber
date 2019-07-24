#!/bin/bash
module load python/3.6-anaconda-4.4
paramfile=../settings/params_lsstall_cmblens_simple_bias_zmax10886.pkl
srun -n 25 python -u clpsiphi_VB_parallel_magbias.py $paramfile
#srun -n 100 python -u clphigal_VB_parallel.py $paramfile
#srun -n 25 python -u clpsiphi_VB_parallel.py $paramfile
