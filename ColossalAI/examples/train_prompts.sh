torchrun --standalone --nproc_per_node=1 examples/train_prompts.py \
         --model 'gpt2' \
         --pretrain "examples/stage1_v2" \
         --rm_pretrain "gpt2" \
         --rm_path "examples/stage2.pt" \
         --strategy naive \
         --prompt_dataset 'examples/prompt.csv' \
         --pretrain_dataset 'examples/instinwild_en.json' \
         --max_epochs 1 \
         --num_episodes 1 \
         --train_batch_size 1 \
         --experience_batch_size 1 \
         --lora_rank 16 \
         --save_path examples/stage3.pt \
	

