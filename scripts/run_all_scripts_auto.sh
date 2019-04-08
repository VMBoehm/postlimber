#!/bin/bash
module load python/3.6-anaconda-4.4

paramfile=../settings/params_gaussgal_z050_51_sigmaz1_1.pkl
srun -n 100 python -u clphiphi_VB_parallel_split2.py $paramfile
srun -n 100 python -u clphiphi_VB_parallel.py $paramfile
srun -n 100 python -u clpsigal_VB_parallel1.py $paramfile
srun -n 100 python -u clpsigal_VB_parallel2.py $paramfile
srun -n 100 python -u clphidelta_VB_parallel_M31b.py $paramfile
paramfile=../settings/params_gaussgal_z060_60_sigmaz1_1.pkl
srun -n 100 python -u clphiphi_VB_parallel_split2.py $paramfile
srun -n 100 python -u clphiphi_VB_parallel.py $paramfile
srun -n 100 python -u clpsigal_VB_parallel1.py $paramfile
srun -n 100 python -u clpsigal_VB_parallel2.py $paramfile
srun -n 100 python -u clphidelta_VB_parallel_M31b.py $paramfile
paramfile=../settings/params_gaussgal_z020_50_sigmaz1_1.pkl
#srun -n 100 python -u clphiphi_VB_parallel_split2.py $paramfile
#srun -n 100 python -u clphiphi_VB_parallel.py $paramfile
srun -n 100 python -u clpsigal_VB_parallel1.py $paramfile
srun -n 100 python -u clpsigal_VB_parallel2.py $paramfile
srun -n 100 python -u clphidelta_VB_parallel_M31b.py $paramfile
