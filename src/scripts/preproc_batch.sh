#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

cd /home/sjne/projects/akdslab-spatial

conda init
conda activate iSCALE

for dist in 100 200 400 800; do
    echo "$dist"
    python -m src.scripts.preprocess -g hex -d 55 -D "$dist" -p /opt/gpudata/sjne/data_for_istar/hex_55_"$dist"
done
