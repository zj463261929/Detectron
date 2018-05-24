
LOG="test.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG" 

python tools/test_net_new.py \
--cfg configs/12_2017_baselines/retinanet_R-101-FPN_1x.yaml \
--multi-gpu-testing \
TEST.WEIGHTS output/oil/retinanet_R-101-FPN_P7_2_R6_S4_O4_Conv4_fre2_ft_my_0.005/train/voc_2007_train/retinanet/model_iter9999.pkl \
NUM_GPUS 2



