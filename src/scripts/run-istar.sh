#!/bin/bash

cd /opt/gpudata/sjne/istar

export CUDA_VISIBLE_DEVICES=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate iSCALE

./run.sh $1/
