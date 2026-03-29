#!/bin/bash

EXPERIMENT_VERSION=$1

echo "SLURM_VISIBLE_DEVICES: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi

REPO="/home/lfranceschi/repos/ACL-SSL"
DATA=$REPO/datasets
SAVE_PATH=$REPO/train_outputs/$SLURM_JOBID

cd $REPO

mkdir -p $SAVE_PATH

set -a; source config/.env; set +a

# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 12345 \
python \
train_ACL.py \
--model_name ADCL_ViT16 \
--model_path $REPO/pretrain \
--exp_name aclifa_2gpu \
--train_config $EXPERIMENT_VERSION \
--vggss_path $DATA/VGGSS \
--flickr_path $DATA/Flickr \
--avs_path $DATA/AVSBench/AVS1 \
--vggsound_path $DATA/vggsound \
--san_path $DATA/silence_and_noise/audio \
--save_path $SAVE_PATH \
--wandb_logging

# --recover_from $REPO/train_outputs/2103685/Train_record/ACL_ViT16_aclifa_2gpu/Param_13.pth \