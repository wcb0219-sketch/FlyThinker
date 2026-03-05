#!/bin/bash

#----------------------------------------------------------------------
# SFT Train and Infer
#----------------------------------------------------------------------

echo "$(date '+%Y-%m-%d %H:%M:%S'): SFT Train"

deepspeed --num_gpus=8 ./Train/train_sft.py

echo "$(date '+%Y-%m-%d %H:%M:%S'): SFT Infer"

torchrun --nproc_per_node=2 ./Infer/infer_sft.py --ckpt_path your_path --output_name infer_sft

#----------------------------------------------------------------------
# FlyThinker Train and Infer
#----------------------------------------------------------------------

echo "$(date '+%Y-%m-%d %H:%M:%S'): Train FlyThinker(G=7B,R=1.5B)"

deepspeed --num_gpus=8 ./Train/train_flythinker.py --diff_lambda 0.5 --save_name output_flythinker

echo "$(date '+%Y-%m-%d %H:%M:%S'): Infer FlyThinker(G=7B,R=1.5B)"

# Non-parallel version (DDP enabled for test).
torchrun --nproc_per_node=2 ./Infer/infer_flythinker_noparallel.py --ckpt_path your_path --diff_lambda 0.5 --output_name infer_flythinker

# Parallel version (no DDP; inference is parallelized per sample to support real-world deployment).
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 ./Infer/infer_flythinker_parallel.py \
  --ckpt_path your_path \
  --lambda 0.5 \
  --output_name infer_flythinker \
  --max_new_tokens 1536 \
  --temperature 0.8 \
  --top_p 0.95
