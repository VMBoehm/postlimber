#!/bin/bash
module load python/3.6-anaconda-4.4

paramfile=../settings/params_auto_gaussgal_z020_20_sigmaz1_1simple_biaszmax25.pkl
#srun -n 100 python -u clphiphi_VB_parallel_split2.py $paramfile
#srun -n 100 python -u clphiphi_VB_parallel.py $paramfile
srun -n 100 python -u clpsigal_VB_parallel1.py $paramfile
srun -n 100 python -u clpsigal_VB_parallel2.py $paramfile
srun -n 100 python -u clphidelta_VB_parallel_M31b.py $paramfile
paramfile=../settings/params_auto_gaussgal_z020_10_sigmaz1_3simple_biaszmax25.pkl
srun -n 100 python -u clpsigal_VB_parallel1.py $paramfile
srun -n 100 python -u clpsigal_VB_parallel2.py $paramfile
srun -n 100 python -u clphidelta_VB_parallel_M31b.py $paramfile
paramfile=../settings/params_auto_gaussgal_z010_10_sigmaz3_1simple_biaszmax25.pkl
srun -n 100 python -u clpsigal_VB_parallel1.py $paramfile
srun -n 100 python -u clpsigal_VB_parallel2.py $paramfile
srun -n 100 python -u clphidelta_VB_parallel_M31b.py $paramfile
paramfile=../settings/params_auto_gaussgal_z010_10_sigmaz2_2simple_biaszmax25.pkl
srun -n 100 python -u clpsigal_VB_parallel1.py $paramfile
srun -n 100 python -u clpsigal_VB_parallel2.py $paramfile
srun -n 100 python -u clphidelta_VB_parallel_M31b.py $paramfile



