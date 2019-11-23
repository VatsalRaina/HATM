#!/bin/bash
#$ -S /bin/bash

export LD_LIBRARY_PATH="/home/mifs/am969/cuda-8.0/lib64:${LD_LIBRARY_PATH}"

export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE

/home/alta/relevance/vr311/attention-topic-model/preprocessing/make_bert_weights.py
