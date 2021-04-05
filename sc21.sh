#!/bin/bash

for (( i=1; i<=32; i*=2 ))
do
    echo "Running with $i processes ..."
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DACE_compiler_use_cache=1 timeout 60s mpirun -n $i python samples/distributed/explicit/sc21_bench.py
done

