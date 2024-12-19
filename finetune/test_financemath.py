import os
import sys

import fire
import torch
import json
import transformers
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Dict
import json
from tqdm import tqdm
from tool import *
from datetime import datetime
import random
from collections import Counter
from model_inputs_utils_finmath import *



def test_convfinqa(base_model,
                   lora_weights,
                   temperature,
                   few_shot,
                   num_return_sequences,
                   max_new_tokens,
                   dataset_path,
                   do_sample,
                   errror_path
                  ):
    
    
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")
    correct, wrong = 0, 0
    filename = f'outputs/finqa_{dt_string}.json'
    use_peft = True
    load_in_8bit=False
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        do_sample=do_sample
    )
    if use_peft:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0},
            do_sample=do_sample
        )
    
    
    #if not load_in_8bit:
    #    model.half() 
    
    
    
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=1,
        do_sample=do_sample,
        pad_token_id = 0,
        num_return_sequences = num_return_sequences,
        #stop_strings = '\n\n'
    )
    
    with open(dataset_path) as f:
        tabmwp = json.load(f)
    #random.shuffle(tabmwp)    
    #jiequ
    #tabmwp = tabmwp[0:3]
    error = []
    for example in tqdm(tabmwp):
        system_input, user_input = prepare_pot_model_input2(example)
        # print(system_input)
        # print("\n\n__________________\n\n")
        # print(user_input)
        full_prompt = system_input + user_input
        #print(full_prompt)
        
        inputs = tokenizer(full_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda") 

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample
            )
        result_counter = Counter()
        codes=[]
        s = generation_output.sequences
        for si in s:
            output = tokenizer.decode(si, skip_special_tokens=True).strip()
            lq = len(full_prompt)
            print(output)
            output = output.split('```python\n')[3]
            output = output.split('```')[0]
            if num_return_sequences > 1:
                output = output.replace('!','')
            codes.append(output)

        #code = output.split('\n\n')[0]
        
        for r in codes:
            #r += '\nans = solution()' 
            print(r)
            ans = safe_execute(r)
            ans = floatify_ans(ans)
            if ans is not None:
                result_counter.update([ans])

        if len(result_counter) > 0:
            prediction = result_counter.most_common(1)[0][0]
        else:
            prediction = None
        
        print(prediction)
        gt_ans = example['ground_truth']
        gt_ans = float(gt_ans)


        if prediction is None:
            wrong += 1
            error.append(example)
        elif finqa_equal(prediction, gt_ans):
            correct += 1
        else:
            wrong += 1
            error.append(example)
        print("prediction:", prediction)
        print("ans:", gt_ans)
        print('accuracy: ', correct / (correct + wrong))
        
        example['generated'] = codes
        
    save_dict_list_to_json(error, errror_path)
    accuracy = correct / (correct + wrong)
    return accuracy
    
if __name__ == "__main__":
    base_model = ["/root/autodl-tmp/llama3-8b"] 
    lora_weights = ["/root/autodl-tmp/finetune/wocode"]
    dataset_path = ['data/validation.json']
    errror_path = ['finqa_error.json']
    temperature = [0.1]
    few_shot = [False] 
    do_sample = [False]
    num_return_sequences = [1]
    max_new_tokens = [256]
    res_list = []
    for i in range(len(base_model)):
        res = test_convfinqa(base_model[i],
                             lora_weights[i],
                             temperature[i],
                             few_shot[i],
                             num_return_sequences[i],
                             max_new_tokens[i],
                             dataset_path[i],
                             do_sample[i],
                             errror_path[i]
                            )
        res_list.append(res)
        print(res_list)