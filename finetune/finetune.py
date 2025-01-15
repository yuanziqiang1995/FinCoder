import os
import sys
from typing import List


import torch
import transformers
from datasets import load_dataset
import json
from tqdm import tqdm
import argparse
import random
os.environ["WANDB_DISABLED"] = "true"



"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from typing import Union, Dict, Any


class Prompter(object):
    
    def generate_prompt(
        self,
        instruction: str,
        label: Union[None, str] = None,
    ) -> str:     

        res = f"{instruction}"
               
        if label:
            res = f"{res}{label}"
         
        return res


    def get_response(self, output: str) -> str:
        return output.split("#Python:\n")[1].strip().replace("/", "\u00F7").replace("*", "\u00D7")
        # return output
        
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    load_peft_weights,
    PeftModel,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

def create_reader_request_processed_finqa(example: Dict[str, Any]):
    prompt = 'Read the following text and table, and then write code to answer a question:\n'
    prompt += 'Please store the result of the question in the \'ans\' variable.\n'
    if example['text']:
        prompt += example['text'] + '\n'
    prompt += example['table'].strip() + '\n'
    prompt += 'Question: {}\n'.format(example['question'])
    prompt += '#Python:\n'
    return prompt

def create_reader_request_processed_convfinqa(example: Dict[str, Any]):
    prompt = 'Read the following text and table, and then write code to answer the last question in a series of questions:\n'
    prompt += 'Please store the result of the last question in a series of questions ths in the \'ans\' variable.\n'
    if example['text']:
        prompt += example['text'].strip() + '\n'
    if example['table']:
        prompt += example['table'].strip() + '\n'
    #prompt += '\n'
    prompt += 'Questions: '
    prompt += " ".join(example['questions'][:-1])
    prompt += '\n'
    prompt += f'Question: {example["questions"][-1]}\n'
    prompt += '#Python:\n'
    return prompt

def create_reader_request_processed_tatqa(example: Dict[str, Any]):
    prompt = 'Read the following text and table, and then write code to answer a question:\n'
    prompt += 'Please store the result of the question in the \'ans\' variable and \'scale\' variable.\n'
    prompt += "If the value of the 'ans' is numerical, predict its scale and store it in a variable named 'scale'. \nThe value of 'scale' can be one of the following: '', 'percent', 'thousand', 'million', or 'billion'. For non-numerical values, set the value of 'scale' to ''"
    if example['text']:
        prompt += example['text'].strip() + '\n'
    prompt += example['table'].strip() + '\n'
    question = example['question']
    prompt += f'Quesetion: {question}\n'
    prompt += '#Python:\n'
    return prompt

def train(
    # model/data params
    base_model: str = "/root/autodl-tmp/llama3-8b",  # Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'
    data_path: str = "/root/autodl-tmp/finetune/data/cft-gpt4-1.json",
    output_dir: str = "./finllms_tatqa",
    data_type: str = "mix", # mix \ finqa\ tatqa \ convfinqa
    # training hyperparams
    batch_size: int = 40,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 3072,
    val_set_size: int = 0, # we don't need val in our case.
    
    # lora hyperparams
    lora_r: int = 160,
    lora_alpha: int = 160,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ],
    
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve

    
    #resume_from_checkpoint: str = "/root/autodl-tmp/finetune/cft-gpt4-e5-1/checkpoint-3700",  # either training checkpoint or final adapter
    resume_from_checkpoint = None
    
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter()

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size


    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0
    
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if data_type == 'mix':
            if data_point['dataset'] == 'finqa':
                ins = create_reader_request_processed_finqa(data_point)
            elif data_point['dataset'] == 'convfinqa':
                ins = create_reader_request_processed_convfinqa(data_point)
            elif data_point['dataset'] == 'tatqa':
                ins = create_reader_request_processed_tatqa(data_point)
            else:
                ins = create_reader_request_processed_finqa(data_point)
        elif data_type == 'finqa':
            ins = create_reader_request_processed_finqa(data_point)
        elif data_type == 'tatqa':
            ins = create_reader_request_processed_tatqa(data_point)
        else:
            ins = create_reader_request_processed_convfinqa(data_point)
        #ins = create_reader_request_processed_finqa(data_point)
        code = data_point["generated"][0]
        if '\nunits = ' in code:
            code = code.replace('\nunits = ', '\nscale = ')
            print(code)
        full_prompt = prompter.generate_prompt(
            ins,
            code,
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                ins
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)
    if resume_from_checkpoint:
        model.enable_input_require_grads()
        model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=True)
        #model._mark_only_adapters_as_trainable()
    else:
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    '''
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
  

    '''
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
    '''
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    '''
    with open(data_path) as f:
        finqa_train = json.load(f)
    train_data = []
    for data_point in tqdm(finqa_train):
        example = generate_and_tokenize_prompt(data_point)
        train_data.append(example)
        
    random.shuffle(train_data)
    
    val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=False if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict


    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    #trainer.train()
    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    train()