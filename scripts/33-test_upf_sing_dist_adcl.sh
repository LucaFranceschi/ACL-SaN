#!/bin/bash

EXPERIMENT_VERSION=$1
JOB_ID=$2
PATH_TO_THRESHOLDS=$3
EPOCHS=$4

echo "SLURM_VISIBLE_DEVICES: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi

REPO="/home/lfranceschi/repos/ACL-SSL"
DATA=$REPO/datasets
SAVE_PATH=$REPO/train_outputs/$JOB_ID/$SLURM_JOBID

cd $REPO

mkdir -p $SAVE_PATH

python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 12345 eval_ACL.py \
--model_name ADCL_ViT16 \
--model_path $REPO/pretrain \
--train_config $EXPERIMENT_VERSION \
--vggss_path $DATA/VGGSS \
--flickr_path $DATA/Flickr \
--avs_path $DATA/AVSBench/AVS1 \
--vggsound_path $DATA/vggsound \
--avatar_path $DATA/AVATAR \
--san_path $DATA/silence_and_noise/audio \
--model_weights $REPO/train_outputs/$JOB_ID \
--path_to_thresholds $PATH_TO_THRESHOLDS \
--epochs $EPOCHS \
--save_path $SAVE_PATH
