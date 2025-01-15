import os
import sys


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
prompt_4shot = """Read the following text and table, and then write code to answer a question:
( in millions ) | dec 282013 | dec 292012
available-for-sale investments | $ 18086 | $ 14001
cash | 854 | 593
equity method investments | 1038 | 992
loans receivable | 1072 | 979
non-marketable cost method investments | 1270 | 1202
reverse repurchase agreements | 800 | 2850
trading assets | 8441 | 5685
total cash and investments | $ 31561 | $ 26302
Question: what percentage of total cash and investments as of dec . 29 2012 was comprised of available-for-sale investments?
#Python
available_for_sale_investments_dec_29_2012 = 14001
total_cash_and_investments_dec_29_2012 = 26302
percent_available_for_sale_investments_dec_29_2012 = available_for_sale_investments_dec_29_2012 / total_cash_and_investments_dec_29_2012
ans = percent_available_for_sale_investments_dec_29_2012


Read the following text and table, and then write code to answer a question:
the chart shows that the firm posted market risk 2013related gains on 248 out of 261 days in this period , with 12 days exceeding $ 210 million .
december 31 ( in millions ) | 1 basis point increase in jpmorgan chase 2019s credit spread
2010 | $ 35
2009 | $ 39
Question: on what percent of trading days were there market gains above $ 210 million?
#Python
days_with_market_gains_above_210_million = 12
total_trading_days = 261
percent_days_with_market_gains_above_210_million = days_with_market_gains_above_210_million / total_trading_days
ans = percent_days_with_market_gains_above_210_million


Read the following text and table, and then write code to answer a question:
american tower corporation and subsidiaries notes to consolidated financial statements ( 3 ) consists of customer-related intangibles of approximately $ 75.0 million and network location intangibles of approximately $ 72.7 million . the customer-related intangibles and network location intangibles are being amortized on a straight-line basis over periods of up to 20 years .
- | preliminary purchase price allocation
current assets | $ 8763
non-current assets | 2332
property and equipment | 26711
intangible assets ( 1 ) | 21079
other non-current liabilities | -1349 ( 1349 )
fair value of net assets acquired | $ 57536
goodwill ( 2 ) | 5998
Question: for acquired customer-related and network location intangibles , what is the expected annual amortization expenses , in millions?
#Python
customer_related_intangibles = 75
network_location_intangibles = 72.7
straight_line_basis = 20
amortization_expenses = ( customer_related_intangibles + network_location_intangibles ) / straight_line_basis
ans = amortization_expenses


Read the following text and table, and then write code to answer a question:
the aggregate commitment under the liquidity asset purchase agreements was approximately $ 23.59 billion and $ 28.37 billion at december 31 , 2008 and 2007 , respectively .
( dollars in billions ) | 2008 amount | 2008 percent of total conduit assets | 2008 amount | percent of total conduit assets
united states | $ 11.09 | 46% ( 46 % ) | $ 12.14 | 42% ( 42 % )
australia | 4.30 | 17 | 6.10 | 21
great britain | 1.97 | 8 | 2.93 | 10
spain | 1.71 | 7 | 1.90 | 7
italy | 1.66 | 7 | 1.86 | 7
portugal | 0.62 | 3 | 0.70 | 2
germany | 0.57 | 3 | 0.70 | 2
netherlands | 0.40 | 2 | 0.55 | 2
belgium | 0.29 | 1 | 0.31 | 1
greece | 0.27 | 1 | 0.31 | 1
other | 1.01 | 5 | 1.26 | 5
total conduit assets | $ 23.89 | 100% ( 100 % ) | $ 28.76 | 100% ( 100 % )
Question: what is percentage change in total conduit asset from 2007 to 2008?
#Python
total_conduit_assets_2007 = 28.76
total_conduit_assets_2008 = 23.89
net_change_in_total_conduit_assets = total_conduit_assets_2008 - total_conduit_assets_2007
percent_change_in_total_conduit_assets = net_change_in_total_conduit_assets / total_conduit_assets_2007
ans = percent_change_in_total_conduit_assets
"""
# def create_reader_request_processed(example: Dict[str, Any]):
#     prompt = 'Read the following text and table, and then write code to answer a question:\n'
#     if example['text']:
#         prompt += example['text'] + '\n'
#     prompt += example['table'].strip() + '\n'
#     prompt += 'Note: The result of the finance Problem should be stored in the \'ans\' variable.\n'
#     prompt += 'Question: {}\n'.format(example['question'])
#     #prompt += '#Python\n\n#Python:\n'
#     prompt += '#Python\n'
#     return prompt

def create_reader_request_processed(example: Dict[str, Any]):
    prompt = 'Read the following text and table, and then write code to answer a question:\n'
    prompt += 'Please store the result of the question in the \'ans\' variable.\n'
    if example['text']:
        prompt += example['text'] + '\n'
    prompt += example['table'].strip() + '\n'
    prompt += 'Question: {}\n'.format(example['question'])
    prompt += '#Python:\n'
    return prompt
    


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
        finqa_dev = json.load(f)
    random.shuffle(finqa_dev)    
    #jiequ
    #finqa_dev = finqa_dev[0:3]
    error = []
    for example in tqdm(finqa_dev):
        if few_shot:
            full_prompt = prompt_4shot + "\n\n"
        else:
            full_prompt = ""
        full_prompt += create_reader_request_processed(example)
        print("example\n"+example['id'])
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
            output = output.split('#Python:\n')[-1]
            output = output.split('\n\n')[0]
            if num_return_sequences > 1:
                output = output.replace('!','')
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
            error.append(example)
        elif finqa_equal(prediction, example['answer'], False):
            correct += 1
        else:
            wrong += 1
            error.append(example)
        print("prediction:", prediction)
        print("ans:", example['answer'])
        print('accuracy: ', correct / (correct + wrong))
        
        example['generated'] = codes
        
    save_dict_list_to_json(error, errror_path)
    accuracy = correct / (correct + wrong)
    return accuracy
    
if __name__ == "__main__":
    base_model = ["/root/autodl-tmp/llama3-8b"] 
    lora_weights = ["/root/autodl-tmp/finetune/finqa_wocode"]
    dataset_path = ['data/finqa_test.json']
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