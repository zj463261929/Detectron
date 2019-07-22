#!/usr/bin/env bash

LOG="res101_1080_jiuquan_coco.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG" 

export CUDA_VISIBLE_DEVICES=0


python tools/test_net_new_recall.py \
--cfg configs/12_2017_baselines/retinanet_101.yaml \
TEST.WEIGHTS output/oil/res101_1080_jiuquan_coco/train/coco_2007_val_jiuquan/retinanet/model_iter4999.pkl 
#NUM_GPUS 1
