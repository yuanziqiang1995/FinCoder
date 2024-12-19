import os
import sys

import fire
import torch
import json
import transformers
from eval_tatqa.tatqa_metric import TaTQAEmAndF1
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
#prompt += "If the value of the `ans` is numerical, predict its units and store it in a variable named `units`. \nThe value of `units` can be one of the following: `none`, `percent`, `thousand`, `million`, or `billion`. For non-numerical values, set the value of `units` to 'none'"
def create_reader_request_processed(example: Dict[str, Any]):
    prompt = 'Read the following text and table, and then write code to answer a question:\n'
    if example['text']:
        prompt += example['text'].strip() + '\n'
    prompt += example['table'].strip() + '\n'
    question = example['question']
    prompt += 'Note: The results should be stored in the ans variable and units variable\n'
    prompt += "If the value of the `ans` is numerical, predict its units and store it in a variable named `units`. \n\nThe value of `units` can be one of the following: `none`, `percent`, `thousand`, `million`, or `billion`. For non-numerical values, set the value of `units` to 'none'"
    prompt += f'Quesetion: {question}\n'
    prompt += '#Python\n'
    return prompt
prompt_1shot = """
"""
prompt_8shot = """
Read the following text and table, and then write code to answer a question:
ASSUMPTIONS USED IN STOCK OPTION PRICING MODEL
The fair value of options granted was determined using a variation of a binomial option pricing model that takes into account factors specific to the share incentive plans, such as the vesting period. The following table shows the principal assumptions used in the valuation.
Expected dividend growth is commensurate with BCE’s dividend growth strategy. Expected volatility is based on the historical volatility of BCE’s share price. The risk-free rate used is equal to the yield available on Government of Canada bonds at the date of grant with a term equal to the expected life of the options
— | 2019 | 2018
Weighted average fair value per option granted | $2.34 | $2.13
Weighted average share price | $58 | $57
Weighted average exercise price | $58 | $56
Expected dividend growth | 5% | 5%
Expected volatility | 14% | 12%
Risk-free interest rate | 2% | 2%
Expected life (years) | 4 | 4
Question: How is the fair value of options granted determined?
#Python
ans = 'using a variation of a binomial option pricing model that takes into account factors specific to the share incentive plans, such as the vesting period'
units = \'\'


Read the following text and table, and then write code to answer a question:
7. Employee numbers and costs
The average monthly number of employees (including Executive Directors but excluding third-party contractors) employed by the Group was as follows:
— | 2019 | 2018
— | Number | Number
Customer operations | 370 | 380
Product and technology | 317 | 312
Corporate | 115 | 130
Total | 802 | 822
Question: What are the categories of employees listed in the table?
#Python
ans = ['Customer operations', 'Product and technology', 'Corporate']
units = \'\'


Read the following text and table, and then write code to answer a question:
Lines of Credit
The following table summarizes our available lines of credit and committed and uncommitted lines of credit, including the revolving credit facility discussed above, and the amounts available under our accounts receivable securitization programs.
(1) Includes total borrowings under the accounts receivable securitization programs, the revolving credit facility and borrowings under lines of credit available to several subsidiaries.
(2) Of the total available lines of credit, $1,137.4 million were committed as of December 31, 2019.
— | December 31, | —
(In millions) | 2019 | 2018
Used lines of credit (1) | $ 98.9 | $ 232.8
Unused lines of credit | 1,245.2 | 1,135.3
Total available lines of credit(2) | $ 1,344.1 | $ 1,368.1
Quesetion: How much was commited as of December 31, 2019 of total available lines of credit?
#Python
ans = '$1,137.4 million'
scale = \'\'


Read the following text and table, and then write code to answer a question:
17. Income Taxes
Income before income taxes for the Company’s domestic and foreign operations was as follows:
— | — | Years Ended June 30, | —
($ in millions) | 2019 | 2018 | 2017
Domestic | $204.2 | $140.3 | $56.0
Foreign | 11.8 | 19.9 | 14.2
Income before income taxes | $216.0 | $160.2 | $70.2
Quesetion: What was the change in Foreign in 2019 from 2018?
#Python
foreign_in_2018 = 19.9
foreign_in_2019 = 11.8
ans = foreign_in_2019 - foreign_in_2018
units = \'million\'


Read the following text and table, and then write code to answer a question:
The following table sets forth the breakdown of revenues by category and segment. Travel revenue includes travel publications (Top 20, Website, Newsflash, Travelzoo Network), Getaway vouchers and hotel platform. Local revenue includes Local Deals vouchers and entertainment offers (vouchers and direct bookings) (in thousands).
Revenue by geography is based on the billing address of the advertiser. Long-lived assets attributed to the U.S. and international geographies are based upon the country in which the asset is located or owned.
Year Ended December 31, | — | —
— | 2019 | 2018
Asia Pacific | — | —
Travel | $6,274 | $7,351
Local | 216 | 508
Total Asia Pacific revenues | 6,490 | 7,859
Europe | — | —
Travel | 32,081 | 30,856
Local | 4,817 | 5,293
Total Europe revenues | 36,898 | 36,149
North America | — | —
Travel | 57,863 | 56,145
Local | 10,161 | 11,169
Total North America revenues | 68,024 | 67,314
Consolidated | — | —
Travel | 96,218 | 94,352
Local | 15,194 | 16,970
Total revenues | $111,412 | 111,322
Question: In 2019, how many geographic regions have total revenues of more than $20,000 thousand?
#Python
total_revenues_in_all_regions = {'Asia Pacific': 6490, 'Europe': 36898, 'North America': 68024}
regions_have_more_than_20000_thousand_total_revenues = [k for k, v in total_revenues_in_all_regions.items() if v > 20000]
ans = len(regions_have_more_than_20000_thousand_total_revenues)
units = \'\'


Read the following text and table, and then write code to answer a question:
11 Intangible assets (continued)
(a) Intangible assets
RIGHTS AND LICENCES
Certain licences that NEXTDC possesses have an indefinite useful life and are carried at cost less impairment losses and are subject to impairment review at least annually and whenever there is an indication that it may be impaired.
Other licences that NEXTDC acquires are carried at cost less accumulated amortisation and accumulated impairment losses. Amortisation is recognised on a straight-line basis over the estimated useful life. The estimated useful life and amortisation method are reviewed at the end of each annual reporting period.
INTERNALLY GENERATED SOFTWARE
Internally developed software is capitalised at cost less accumulated amortisation. Amortisation is calculated using the straight-line basis over the asset’s useful economic life which is generally two to three years. Their useful lives and potential impairment are reviewed at the end of each financial year.
SOFTWARE UNDER DEVELOPMENT
Costs incurred in developing products or systems and costs incurred in acquiring software and licenses that will contribute to future period financial benefits through revenue generation and/or cost reduction are capitalised to software and systems. Costs capitalised include external direct costs of materials and services and employee costs.
Assets in the course of construction include only those costs directly attributable to the development phase and are only recognised following completion of technical feasibility and where the Group has an intention and ability to use the asset.
— | Rights and licenses | Internally generated software | Software under development | Total
Movements | $'000 | $'000 | $'000 | $'000
At 30 June 2019 | — | — | — | —
Cost | 13 | 12,961 | 16,284 | 29,259
Accumulated amortisation | - | -5,580 | - | -5,580
Netbook amount | 13 | 7,381 | 16,284 | 23,678
30 June 2018 | — | — | — | —
Opening net book amount at 1 July 2017 | 43 | 442 | 8,053 | 8,538
Additions – externally acquired | 13 | - | 5,253 | 5,266
Additions – internally developed | - | - | 1,256 | 1,256
Amortisation | -43 | -1,746 | - | -1,789
Transfers | - | 7,563 | -7,563 | -
Transfer between classes | - | 744 | - | 744
Disposals | - | -618 | -490 | -1,108
Closing net book amount | 13 | 6,385 | 6,509 | 12,907
At 30 June 2018 | — | — | — | —
Cost | 104 | 9,555 | 6,509 | 16,168
Accumulated amortisation | -91 | -3,170 | - | -3,261
Net book amount | 13 | 6,385 | 6,509 | 12,907
Quesetion: Which year have greater total accumulated amortisation?
#Python
total_accumulated_amortisation = {'2019': 5580, '2018': 3261}
ans = sorted(total_accumulated_amortisation.items(), key=lambda tup: tup[1], reverse=True)[0][0]
units = \'year\'


Read the following text and table, and then write code to answer a question:
Effective Income Tax Rate
A reconciliation of the United States federal statutory income tax rate to our effective income tax rate is as follows:
In 2019 and 2018 we had pre-tax losses of $19,573 and $25,403, respectively, which are available for carry forward to offset future taxable income. We made determinations to provide full valuation allowances for our net deferred tax assets at the end of 2019 and 2018, including NOL carryforwards generated during the years, based on our evaluation of positive and negative evidence, including our history of operating losses and the uncertainty of generating future taxable income that would enable us to realize our deferred tax.
— | Year Ended | Year Ended
— | December 31, 2018 | December 31, 2019
United States federal statutory rate | 21.00% | 21.00%
State taxes, net of federal benefit | 1.99% | -0.01%
Valuation allowance | -21.96% | -24.33%
Cumulative effect of accounting change | — | 2.07%
R&D Credit | 1.34% | 1.53%
Other | -0.38% | -0.27%
Effective income tax rate | 1.99% | -0.01%
Quesetion: What was the 2019 percentage change in pre-tax losses?
#Python
pre_tax_losses_2018 = 25403
pre_tax_losses_2019 = 19573
net_change = pre_tax_losses_2019 - pre_tax_losses_2018
ans = net_change / pre_tax_losses_2018 * 100
units = \'percent\'
"""
    

def test_tatqa(base_model,
               lora_weights,
               temperature,
               few_shot,
               num_return_sequences,
               max_new_tokens,
               dataset_path,
               do_sample,
               output_path
              ):

    
    
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")
    correct, wrong = 0, 0
    filename = f'outputs/tatqa_{dt_string}.json'
    use_peft = True
    load_in_8bit=False

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    em_and_f1 = TaTQAEmAndF1()
    
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
        num_return_sequences = num_return_sequences
    )
    with open(dataset_path) as f:
        finqa_dev = json.load(f)
    random.shuffle(finqa_dev)    
    #jiequ
    #finqa_dev = finqa_dev[0:3]
    result = {}
    for example in tqdm(finqa_dev):
        #full_prompt = prompt_4shot + "\n\n"
        full_prompt = ""
        if few_shot:
            full_prompt = prompt_8shot + "\n\n"
        full_prompt += create_reader_request_processed(example)
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

        codes=[]
        s = generation_output.sequences
        
        answer_counter = Counter()
        units_counter = Counter()
        
        for si in s:
            output = tokenizer.decode(si, skip_special_tokens=True).strip()
            lq = len(full_prompt)
            print(output)
            output = output.split('#Python\n')[1]
            output = output.split('\n\n')[0]
            #output = output.replace('!','')
            codes.append(output)
        print(codes)
        #code = output.split('\n\n')[0]
        
        for r in codes:
            ans = safe_execute(r, keys=['ans', 'units'])

            if ans is None:
                ans, units = None, None
            else:
                ans, units = ans
                if units == None:
                    units = ''
            if ans is not None:
                answer_counter.update([str(ans)])
            if units is not None:
                units_counter.update([str(units)])

        if len(answer_counter) > 0:
            pred_answer = answer_counter.most_common(1)[0][0]
            try:
                pred_answer = eval(pred_answer)
            except:
                pred_answer = pred_answer
        else:
            pred_answer = ''

        if len(units_counter) > 0:
            pred_scale = units_counter.most_common(1)[0][0]
        else:
            pred_scale = ''


        # Furthe Process according to TATQA dataset
        if type(pred_answer) == str:
            pred_answer = [pred_answer]
        if type(pred_answer) == list and len(pred_answer) < 1:
            pred_answer = ''
        if type(pred_answer) == list and type(pred_answer[0]) == str:
            if pred_scale and pred_scale in pred_answer[0]:
                pred_scale = ''
        result[example['question_id']] = [pred_answer, pred_scale]
        print('pred\n')
        print(pred_answer, pred_scale)
    save_dict_list_to_json(result, output_path)

if __name__ == "__main__":
    base_model = ["/root/autodl-tmp/llama3-8b"]
    lora_weights = ["/root/autodl-tmp/finetune/weights"]
    dataset_path = ['data/tatqa_dev.json']
    output_path = ['data/tatqa_prediction_dev_test']
    temperature = [0.1]
    few_shot = [False]
    do_sample = [False]
    num_return_sequences = [1]
    max_new_tokens = [256]
    res_list = []
    for i in range(len(base_model)):
        test_tatqa(base_model[i],
                   lora_weights[i],
                   temperature[i],
                   few_shot[i],
                   num_return_sequences[i],
                   max_new_tokens[i],
                   dataset_path[i],
                   do_sample[i],
                   output_path[i],
                  )
