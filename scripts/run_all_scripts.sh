#!/bin/bash
paramfile=../settings/params_lsstall_cmblens.pkl
srun -n 100 python -u clphiphi_VB_parallel_split.py $paramfile
srun -n 100 python -u clphiphi_VB_parallel_split2.py $paramfile
srun -n 100 python -u clphiphi_VB_parallel.py $paramfile
srun -n 25 python -u clphiphi_VB_parallel_source.py $paramfile
srun -n 100 python -u clphigal_VB_parallel.py $paramfile
srun -n 100 python -u clphidelta_VB_parallel_MB2.py $paramfile
srun -n 100 python -u clpsiphi_VB_parallel.py $paramfile
