CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 infer_flythinker_noparallel.py \
        --ckpt_path your_ckpt_path \
        --lambda_diff 0.5 \
        --output_name flythinker_noparallel