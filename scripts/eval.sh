#!/bin/bash
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
# export GLOG_minloglevel=2
MASTER_PORT=$((RANDOM % 101 + 20000))

cuda_visible_devices=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=${cuda_visible_devices}
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

CHECKPOINT="/home/lunet/cohw2/Projects/Test/checkpoints/r2r_rxr" 
echo "CHECKPOINT: ${CHECKPOINT}"
OUTPUT_PATH="eval_results/r2r_rxr2"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
CONFIG="config/vln_r2r.yaml"
echo "CONFIG: ${CONFIG}"
mkdir -p ${OUTPUT_PATH}

export LD_LIBRARY_PATH=/home/lunet/cohw2/.conda/envs/qwen/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

/home/lunet/cohw2/.conda/envs/qwen/bin/torchrun --nproc_per_node=8 \
        --master_port=$MASTER_PORT \
        model/eval.py \
        --model_path $CHECKPOINT \
        --habitat_config_path $CONFIG \
        --output_path $OUTPUT_PATH \
        --eval_split val_unseen \
        --save_video \
        > ${OUTPUT_PATH}/eval.log 2>&1


