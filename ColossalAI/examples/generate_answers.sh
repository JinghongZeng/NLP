torchrun --standalone --nproc_per_node=1 examples/generate_answers.py \
    --model 'gpt2' \
    --strategy 'naive' \
    --model_path "gpt2-medium" \
    --dataset 'examples/question_all.json' \
    --batch_size 1 \
    --max_datasets_size 80 \
    --answer_path 'examples/answers_v2_pre.json' \
    --max_length 128 \

