#!/bin/bash

EXPERIMENT_VERSION=$1
PATH_TO_MODEL=$2
PATH_TO_THRESHOLDS=$3
EPOCHS=$4

echo "SLURM_VISIBLE_DEVICES: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi

REPO="/home/lfranceschi/repos/ACL-SSL"
DATA=$REPO/datasets
SAVE_PATH=$REPO/train_outputs/$SLURM_JOBID

cd $REPO

mkdir -p $SAVE_PATH

python eval_ACL_noisy.py \
--model_name ACL_ViT16 \
--model_path $REPO/pretrain \
--train_config $EXPERIMENT_VERSION \
--vggss_path $DATA/VGGSS \
--flickr_path $DATA/Flickr \
--avs_path $DATA/AVSBench/AVS1 \
--vggsound_path $DATA/vggsound \
--avatar_path $DATA/AVATAR \
--san_path $DATA/silence_and_noise/audio \
--model_weights $PATH_TO_MODEL \
--path_to_thresholds $PATH_TO_THRESHOLDS \
--epochs $EPOCHS \
--save_path $SAVE_PATH
