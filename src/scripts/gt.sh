#!/bin/bash

cd /home/sjne/projects/akdslab-spatial

export CUDA_VISIBLE_DEVICES="$1"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial

echo "$2"
python -m src.scripts.preprocess -g sq -s "$2" -p /opt/gpudata/sjne/data_for_istar/sq_"$2"
