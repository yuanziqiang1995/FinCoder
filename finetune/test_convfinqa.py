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
prompt_4shot = """Read the following text and table, and then write code to answer the last question in a series of questions:
- | shares available for awards | shares subject to outstanding awards
2009 global incentive plan | 2322450 | 2530454
2004 stock incentive plan | - | 5923147
Questions: how many shares are subject to outstanding awards is under the 2009 global incentive plan? what about under the 2004 stock incentive plan? how many total shares are subject to outstanding awards? what about under the 2004 stock incentive plan?
Question: what proportion does this represent?
#Python
shares_subject_to_outstanding_awards_2009_global_incentive_plan = 2530454
shares_subject_to_outstanding_awards_2004_stock_incentive_plan = 5923147
total_shares_subject_to_outstanding_awards = shares_subject_to_outstanding_awards_2009_global_incentive_plan + shares_subject_to_outstanding_awards_2004_stock_incentive_plan
proportion = shares_subject_to_outstanding_awards_2009_global_incentive_plan / total_shares_subject_to_outstanding_awards
ans = proportion


Read the following text and table, and then write code to answer the last question in a series of questions:
compensation expense the company recorded $ 43 million , $ 34 million , and $ 44 million of expense related to stock awards for the years ended december 31 , 2015 , 2014 , and 2013 , respectively . 
Questions: what is the compensation expense the company recorded in 2015? what about in 2014? what is the total compensation expense the company recorded in 2015 and 2014? what is the total expenses including 2013?
Question: what is the average for three years?
#Python
compensation_expense_2015 = 43
compensation_expense_2014 = 34
compensation_expense_2013 = 44
total_compensation_expense = compensation_expense_2015 + compensation_expense_2014 + compensation_expense_2013
average_for_three_years = total_compensation_expense / 3
ans = average_for_three_years


Read the following text and table, and then write code to answer the last question in a series of questions:
the net loss on disposal of those assets was $ 344000 for 2005 and $ 43000 for 2004 . 
Questions: what was the net loss on disposal of assets in 2005? what was the value in 2004? what was the change in value?
Question: what was the percent change?
#Python
net_loss_on_disposal_of_assets_2005 = 344000
net_loss_on_disposal_of_assets_2004 = 43000
net_change_in_value = net_loss_on_disposal_of_assets_2005 - net_loss_on_disposal_of_assets_2004
percent_change = net_change_in_value / net_loss_on_disposal_of_assets_2004
ans = percent_change

Read the following text and table, and then write code to answer the last question in a series of questions:
location | operations conducted | approximatesquare feet | leaseexpirationdates
dublin ireland | global supply chain distribution and administration offices | 160000 | owned
athlone ireland | commercial research and development manufacturing | 80000 | owned
bogart georgia | commercial research and development manufacturing | 70000 | owned
smithfield rhode island | commercial research and development manufacturing | 67000 | owned
Questions: what is the square feet of the owned global supply chain distribution and administration offices? what is the square feet of the owned commercial research and development manufacturing? what is the sum of those values? what is the total sum including square feet of commercial research and development manufacturing in bogart, georgia? what is the total sum including square feet of commercial research and development manufacturing in smithfield, rhode island?
Question: what is the total sum of square feet owned?
#Python
square_feet_owned = [160000, 80000, 70000, 67000]
total_square_feet_owned = sum(square_feet_owned)
ans = total_square_feet_owned"""
def create_reader_request_processed(example: Dict[str, Any]):
    prompt = 'Read the following text and table, and then write code to answer the last question in a series of questions:\n'
    if example['text']:
        prompt += example['text'].strip() + '\n'
    if example['table']:
        prompt += example['table'].strip() + '\n'
    prompt += 'Note: The results should be stored in the \'ans\' variable.\n'
    #prompt += '\n'
    prompt += 'Questions: '
    prompt += " ".join(example['questions'][:-1])
    prompt += '\n'
    prompt += f'Question: {example["questions"][-1]}\n'
    prompt += '#Python\n'
    return prompt
    


if __name__ == "__main__":
    
    
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")
    correct, wrong = 0, 0
    filename = f'outputs/convfinqa_{dt_string}.json'
    use_peft = True
    load_in_8bit=False
    shot_num = 4
    base_model = "/root/autodl-tmp/llama3-8b"
    lora_weights = "/root/autodl-tmp/finetune/weights"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    do_sample = True
    
    
    
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
        temperature=0.1,
        top_p=1,
        do_sample=do_sample,
        pad_token_id = 0,
        num_return_sequences = 1,
        stop_strings = '\n\n'
    )
    max_new_tokens = 256
    with open('data/convfinqa_dev4.json') as f:
        finqa_dev = json.load(f)
    random.shuffle(finqa_dev)    
    #jiequ
    #finqa_dev = finqa_dev[0:3]
    result = []
    for example in tqdm(finqa_dev):
        #full_prompt = prompt_4shot + "\n\n"
        full_prompt = ""
        #full_prompt = prompt_4shot + "\n\n"
        full_prompt += create_reader_request_processed(example)
        #print(full_prompt)
        print("example\n"+example['id'])
        inputs = tokenizer(full_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda") 
        '''
        try:
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample
                )
            s = generation_output.sequences[0]
            output = tokenizer.decode(s, skip_special_tokens=True).strip()
        
            lq = len(full_prompt)
            output = output[lq:]
            code = output.split('\n\n')[0]
            ans = safe_execute(code)
            ans = floatify_ans(ans)
            prediction = ans
        except:
            prediction = None
        '''
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
            output = output.split('#Python\n')[-1]
            output = output.split('\n\n')[0]
            #output = output.replace('!','')
            codes.append(output)
        print(codes)
        #code = output.split('\n\n')[0]
        
        for r in codes:
            ans = safe_execute(r)
            ans = floatify_ans(ans)
            if ans is not None:
                result_counter.update([ans])
        
        if len(result_counter) > 0:
            prediction = result_counter.most_common(1)[0][0]
        else:
            prediction = None

        print(prediction)
        if prediction is not None:
            if type(prediction) == bool:
                if prediction:
                    prediction = 'yes'
                else:
                    prediction = 'no'
            elif type(prediction) == list:
                prediction = prediction[0]
                if type(prediction) == bool:
                    if prediction:
                        prediction = 'yes'
                    else:
                        prediction = 'no'
            else:
                if type(prediction) not in [float, int]:
                    prediction = None

        if prediction is None:
            wrong += 1
        elif finqa_equal(prediction, example['answer'], False):
            correct += 1
        else:
            wrong += 1
        print("prediction:", prediction)
        print("ans:", example['answer'])
        print('accuracy: ', correct / (correct + wrong))
        
        example['generated'] = codes
        
    save_dict_list_to_json(finqa_dev, filename)