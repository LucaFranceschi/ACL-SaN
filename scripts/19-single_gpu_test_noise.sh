#!/bin/bash

EXPERIMENT_VERSION=$1
PATH_TO_MODEL=$2

JOB_ID=$(sed -E 's/.*\/([0-9]+)\/.*/\1/' <<< "$PATH_TO_MODEL")

echo "SLURM_VISIBLE_DEVICES: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi

REPO="/home/lfranceschi/repos/ACL-SSL"
DATA=$REPO/datasets
SAVE_PATH=$REPO/train_outputs/$JOB_ID/$SLURM_JOBID

cd $REPO

mkdir -p $SAVE_PATH

set -a; source config/.env; set +a

python eval_noisy_audio_samples.py \
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
--save_path $SAVE_PATH