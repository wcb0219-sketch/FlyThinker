CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 infer_flythinker_parallel.py \
  --ckpt_path your_ckpt_path \
  --lambda_diff 0.5 \
  --output_name flythinker_parallel \
  --max_new_tokens 1536 \
  --temperature 0.8 \
  --top_p 0.95