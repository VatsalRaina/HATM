#!/bin/bash
#$ -S /bin/bash

export CUDAPATH="/home/mifs/am969/cuda-8.0/targets/x86_64-linux/lib/:/home/mifs/am969/cuda-8.0/lib64:/home/mifs/ar527/bin/OpenBLAS"

qsub -cwd -j y -o LOGs/LOG.prep -l qp=cuda-low -l osrel='*' -l mem_grab=160G -l gpuclass='kepler' ${1}

