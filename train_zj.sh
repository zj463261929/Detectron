#!/usr/bin/env bash

LOG="log/oil/res101_1080_104832.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG" 

#export CUDA_VISIBLE_DEVICES=0

python tools/train_net.py  \
--cfg configs/12_2017_baselines/retinanet_101.yaml \
OUTPUT_DIR output/oil/res101_1080_104832/

