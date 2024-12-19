from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset, Dataset
from trl import SFTTrainer, setup_chat_format
from typing import Union, Dict, Any
import json
os.environ["WANDB_DISABLED"] = "true"
base_model = "/root/autodl-tmp/llama3-8b"
data_path = "/root/autodl-tmp/finetune/data/cft-gpt4-1.json"
new_model = "ins-coder"


def create_reader_request_processed_finqa(example: Dict[str, Any]):
    prompt = 'Read the following text and table, and then write code to answer a question:\n'
    if example['text']:
        prompt += example['text'] + '\n'
    prompt += example['table'].strip() + '\n'
    prompt += 'Note: The result of the finance Problem should be stored in the \'ans\' variable.\n'
    prompt += 'Question: {}\n'.format(example['question'])
    prompt += '#Python\n'
    return prompt

def create_reader_request_processed_convfinqa(example: Dict[str, Any]):
    prompt = 'Read the following text and table, and then write code to answer the last question in a series of questions:\n'
    if example['text']:
        prompt += example['text'].strip() + '\n'
    if example['table']:
        prompt += example['table'].strip() + '\n'
    prompt += 'Note: The results should be stored in the \'ans\' variable.\n'
    prompt += 'Questions: '
    prompt += " ".join(example['questions'][:-1])
    prompt += '\n'
    prompt += f'Question: {example["questions"][-1]}\n'
    prompt += '#Python\n'
    return prompt

def create_reader_request_processed_tatqa(example: Dict[str, Any]):
    prompt = 'Read the following text and table, and then write code to answer a question:\n'
    if example['text']:
        prompt += example['text'].strip() + '\n'
    prompt += example['table'].strip() + '\n'
    question = example['question']
    prompt += 'Note: The results should be stored in the \'ans\' variable and \'units\' variable.\n'
    prompt += f'Quesetion: {question}\n'
    prompt += '#Python\n'
    return prompt


torch_dtype = torch.float16
attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    torch_dtype=torch.float16,
)

device_map="auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    #load_in_8bit=True,
    quantization_config=bnb_config,
    device_map=device_map,
    #attn_implementation=attn_implementation
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)

#Importing the dataset
with open(data_path) as f:
    finqa_train = json.load(f)

def format_chat_template(row):
    row_json = [
        {"role": "system","content": "You are a financial assistant. Always provide accurate and reliable information to the best of your abilities"},
        {"role": "user", "content": row['ins']},
        {"role": "assistant", "content": row['code']}
    ]
    row = {}
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = []
for data_point in finqa_train:
    if data_point['dataset'] == 'finqa':
        ins = create_reader_request_processed_finqa(data_point)
    elif data_point['dataset'] == 'convfinqa':
        ins = create_reader_request_processed_convfinqa(data_point)
    elif data_point['dataset'] == 'tatqa':
        ins = create_reader_request_processed_tatqa(data_point)
    else:
        ins = create_reader_request_processed_finqa(data_point)
    code = data_point["generated"][0]
    d = {'ins':ins, 'code':code}
    dataset.append(d)
    
dataset = Dataset.from_list(dataset) 
dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)

#print(dataset[0])
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="adamw_torch",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    group_by_length=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    #eval_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=3072,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

trainer.train()