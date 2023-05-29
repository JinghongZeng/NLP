import argparse
import os
import random
import copy
import math
from tqdm import tqdm

import torch
import torch.distributed as dist
import transformers

from coati.models.bloom import BLOOMActor
from coati.models.gpt import GPTActor
from coati.models.opt import OPTActor
from coati.models.roberta import RoBERTaActor
from coati.models.llama import LlamaActor
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from transformers import AutoTokenizer, RobertaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from utils import jload, jdump, is_rank_0


PROMPT_DICT = {
    "prompt_input":
        ("Below is an instruction that describes a task, paired with an input that provides further context. "
         "Write a response that appropriately completes the request.\n\n"
         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"),
    "prompt_no_input": ("Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"),
}


def generate(args):
    # torch.cuda.set_per_process_memory_fraction(0.4)
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2_cpu':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cpu')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    with strategy.model_init_context():		
        if args.model == 'gpt2':
            #actor = GPTActor(pretrained=args.model).to(torch.cuda.current_device())
            actor = GPTActor(pretrained=args.model_path).to(torch.cuda.current_device()) # for stage1 model, use this line
        elif args.model == 'bloom':
            actor = BLOOMActor(pretrained=args.model_path).to(
                torch.cuda.current_device())
        elif args.model == 'opt':
            actor = OPTActor(pretrained=args.model_path).to(
                torch.cuda.current_device())
        elif args.model == 'roberta':
            actor = RoBERTaActor(pretrained=args.model_path).to(
                torch.cuda.current_device())
        elif args.model == 'llama':
            actor = LlamaActor(pretrained=args.model_path).to(torch.float16).to(torch.cuda.current_device()) # for stage 1
        else:
            raise ValueError(f'Unsupported model "{args.model}"')

    #state_dict = torch.load(args.model_path, map_location='cpu')
    #actor.load_state_dict(state_dict) 
    # for stage1 model, comment the above two lines and change args.model_path in the above chunk

    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m')
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif args.model == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                                  padding_side="right",
                                                  use_fast=False)
        tokenizer.eos_token = '<\s>'
    else:
        raise ValueError(f'Unsupported model "{args.model}"') 
   
    questions = []
    if args.max_datasets_size is not None:
        questions = random.sample(jload(args.dataset), args.max_datasets_size)

    answers = copy.deepcopy(questions)
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get(
            "input", "") != "" else prompt_no_input.format_map(example)
        for example in questions
    ]

    input_ids_list = []
    for string in sources:
        input_ids = tokenizer.encode(string, return_tensors='pt').squeeze(0)
        input_ids_list.append(input_ids)
    bar = tqdm(range(math.ceil(len(input_ids_list)/args.batch_size)),
               desc=f'steps', disable=not is_rank_0())
    actor.eval()
    with torch.no_grad():
        for i in range(0, len(input_ids_list), args.batch_size):
            batch = input_ids_list[i:i+args.batch_size]
            batch = [i.flip(dims=[0]) for i in batch]
            batch = torch.nn.utils.rnn.pad_sequence(batch,
                                                    batch_first=True,
                                                    padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0).to(torch.cuda.current_device())
            batch = batch.flip(dims=[1])
            attention_mask = batch.ne(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            outputs = actor.model.generate(batch, attention_mask=attention_mask,
                                           max_length=args.max_length,
                                           do_sample=True,
                                           top_k=100, top_p = 0.9,
                                           num_return_sequences=1, num_beams = 5)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for j in range(batch.size(0)):
                answers[i +
                        j]['output'] = outputs[j].split("### Response:")[1].strip()
            bar.update()
    jdump(answers, args.answer_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini',
                                 'colossalai_zero2', 'colossalai_zero2_cpu'],
                        default='naive')
    parser.add_argument('--model', default='gpt2',
                        choices=['gpt2', 'bloom', 'opt', 'roberta', 'llama'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_datasets_size', type=int, default=None)
    parser.add_argument('--answer_path', type=str, default="answer")
    parser.add_argument('--max_length', type=int, default=1024)
    args = parser.parse_args()
    generate(args)
