#!/bin/bash


echo '\ncl13j\n'
time mpirun -n 10 python -u cl13j.py

echo '\ncl13a\n'
time mpirun -n 10 python -u cl13a-phifac.py

echo '\ncl13b\n'
time mpirun -n 10 python -u cl13b.py

echo '\ncl22A\n'
time mpirun -n 10 python -u cl22A-phiinterp.py

echo '\ncl22B\n'
time mpirun -n 10 python -u cl22B.py

echo '\ncl31aA\n'
time mpirun -n 10 python -u cl31aA-phifac.py

echo '\ncl31aB\n'
time mpirun -n 10 python -u cl31aB.py

echo '\ncl31b\n'
time mpirun -n 10 python -u cl31b.py

