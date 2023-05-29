torchrun --standalone --nproc_per_node=1 examples/train_reward_model.py \
   --pretrain  examples/stage1_v2 \
   --model 'gpt2' \
   --strategy naive \
   --loss_fn 'log_exp' \
   --save_path "examples/stage2_v2.pt" \
   --dataset 'Anthropic/hh-rlhf' \
   --max_epochs 1 \
   --batch_size 1 \
   --lora_rank 256 \
