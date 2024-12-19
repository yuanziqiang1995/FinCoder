import json
from time import sleep
from tqdm import tqdm
import os
import openai
from datetime import datetime
from tool import *
from typing import Dict, Any
import argparse
from collections import Counter


input_json_file = 'data/finqa/finqa_train_step1.json'
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
openai.api_key = 'sk-wDzxPXsOAAHaTSLUriUdNSTVIU8srmMRiiKoCg3wC2s42ytj'
openai.api_base = "https://xiaoai.plus/v1"
parser = argparse.ArgumentParser()
parser.add_argument("--key", default='sk-wDzxPXsOAAHaTSLUriUdNSTVIU8srmMRiiKoCg3wC2s42ytj', type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--greedy", default=True, action='store_true')
parser.add_argument("--dry_run", default=False, action='store_true')
parser.add_argument("--end", default=-1, type=int)
args = parser.parse_args()


def create_reader_request_processed(example: Dict[str, Any]):
    prompt = 'Please read the following text and table, and write code to answer the question based on the calculation process:\n'
    if example['text']:
        prompt += example['text'] + '\n'
    prompt += example['table'].strip() + '\n'
    prompt += 'Calculation Process:: {}\n'.format(example['program'])
    prompt += 'Question: {}\n'.format(example['question'])
    prompt += 'Note: The generated code does not contain any comments; The results are stored in the ans variable and do not require printing out\n'
    prompt += '#Python\n'
    prompt += example['generated'][0]
    return prompt


prompt_1shot = """Please read the following text and table, and write code to answer the question based on the calculation process:
( in millions ) | dec 282013 | dec 292012
available-for-sale investments | $ 18086 | $ 14001
cash | 854 | 593
equity method investments | 1038 | 992
loans receivable | 1072 | 979
non-marketable cost method investments | 1270 | 1202
reverse repurchase agreements | 800 | 2850
trading assets | 8441 | 5685
total cash and investments | $ 31561 | $ 26302
Calculation Process: ans = 14001 / 26302
Question: what percentage of total cash and investments as of dec . 29 2012 was comprised of available-for-sale investments?
#Python
available_for_sale_investments_dec_29_2012 = 14001
total_cash_and_investments_dec_29_2012 = 26302
percent_available_for_sale_investments_dec_29_2012 = available_for_sale_investments_dec_29_2012 / total_cash_and_investments_dec_29_2012
ans = percent_available_for_sale_investments_dec_29_2012"""

prompt_4shot_c = """Read the following text and table, and then answer the last question in a series of questions:
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


Read the following text and table, and then answer the last question in a series of questions:
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


Read the following text and table, and then answer the last question in a series of questions:
the net loss on disposal of those assets was $ 344000 for 2005 and $ 43000 for 2004 . 
Questions: what was the net loss on disposal of assets in 2005? what was the value in 2004? what was the change in value?
Question: what was the percent change?
#Python
net_loss_on_disposal_of_assets_2005 = 344000
net_loss_on_disposal_of_assets_2004 = 43000
net_change_in_value = net_loss_on_disposal_of_assets_2005 - net_loss_on_disposal_of_assets_2004
percent_change = net_change_in_value / net_loss_on_disposal_of_assets_2004
ans = percent_change


Read the following text and table, and then answer the last question in a series of questions:
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
ans = total_square_feet_owned
"""

prompt_4shot = """Please read the following text and table, and write code to answer the questions based on the calculation process:
( in millions ) | dec 282013 | dec 292012
available-for-sale investments | $ 18086 | $ 14001
cash | 854 | 593
equity method investments | 1038 | 992
loans receivable | 1072 | 979
non-marketable cost method investments | 1270 | 1202
reverse repurchase agreements | 800 | 2850
trading assets | 8441 | 5685
total cash and investments | $ 31561 | $ 26302
Calculation Process: ans = 14001 / 26302
Question: what percentage of total cash and investments as of dec . 29 2012 was comprised of available-for-sale investments?
#Python
available_for_sale_investments_dec_29_2012 = 14001
total_cash_and_investments_dec_29_2012 = 26302
percent_available_for_sale_investments_dec_29_2012 = available_for_sale_investments_dec_29_2012 / total_cash_and_investments_dec_29_2012
ans = percent_available_for_sale_investments_dec_29_2012


Please read the following text and table, and write code to answer the questions based on the calculation process:
the chart shows that the firm posted market risk 2013related gains on 248 out of 261 days in this period , with 12 days exceeding $ 210 million .
december 31 ( in millions ) | 1 basis point increase in jpmorgan chase 2019s credit spread
2010 | $ 35
2009 | $ 39
Calculation Process: ans = 12 / 261
Question: on what percent of trading days were there market gains above $ 210 million?
#Python
days_with_market_gains_above_210_million = 12
total_trading_days = 261
percent_days_with_market_gains_above_210_million = days_with_market_gains_above_210_million / total_trading_days
ans = percent_days_with_market_gains_above_210_million


Please read the following text and table, and write code to answer the questions based on the calculation process:
american tower corporation and subsidiaries notes to consolidated financial statements ( 3 ) consists of customer-related intangibles of approximately $ 75.0 million and network location intangibles of approximately $ 72.7 million . the customer-related intangibles and network location intangibles are being amortized on a straight-line basis over periods of up to 20 years .
- | preliminary purchase price allocation
current assets | $ 8763
non-current assets | 2332
property and equipment | 26711
intangible assets ( 1 ) | 21079
other non-current liabilities | -1349 ( 1349 )
fair value of net assets acquired | $ 57536
goodwill ( 2 ) | 5998
Calculation Process: x0 = 75 + 72.7; ans = x0 / 20
Question: for acquired customer-related and network location intangibles , what is the expected annual amortization expenses , in millions?
#Python
customer_related_intangibles = 75
network_location_intangibles = 72.7
straight_line_basis = 20
amortization_expenses = ( customer_related_intangibles + network_location_intangibles ) / straight_line_basis
ans = amortization_expenses


Please read the following text and table, and write code to answer the questions based on the calculation process:
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
Calculation Process: x0 = 23.89 - 28.76; ans = x0 / 28.76
Question: what is percentage change in total conduit asset from 2007 to 2008?
#Python
total_conduit_assets_2007 = 28.76
total_conduit_assets_2008 = 23.89
net_change_in_total_conduit_assets = total_conduit_assets_2008 - total_conduit_assets_2007
percent_change_in_total_conduit_assets = net_change_in_total_conduit_assets / total_conduit_assets_2007
ans = percent_change_in_total_conduit_assets
"""
# current 396
if __name__ == "__main__":
    with open(input_json_file) as f:
        finqa_dev = json.load(f)
    finqa_dev = finqa_dev[args.start:]
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")

    correct, wrong = 0, 0
    correct_examples = f'outputs/finqa_correct_s{args.start}_e{args.end}_{dt_string}.jsonl'
    if args.greedy:
        filename = f'outputs/finqa_new_s{args.start}_e{args.end}_{dt_string}.jsonl'
    else:
        filename = f'outputs/finqa_new_sc_s{args.start}_e{args.end}_{dt_string}.jsonl'
    writer = open(filename, 'w')
    writer_c = open(correct_examples, 'w')
    for example in tqdm(finqa_dev):
        full_prompt = prompt_4shot + "\n\n"
        full_prompt += create_reader_request_processed(example)
        # print(full_prompt)
        if args.dry_run:
            print(full_prompt)
            print('=======================')
            break

        if args.greedy:
            # greedy decoding
            got_result = False
            while not got_result:
                try:
                    '''result = openai.Completion.create(
                        engine='code-davinci-002',
                        prompt=full_prompt,
                        api_key=os.getenv(args.key),
                        max_tokens=512,
                        temperature=0.0,
                        top_p=1,
                        n=1,
                        stop=['\n\n'],
                        logprobs=1
                    )'''
                    result = openai.ChatCompletion.create(
                        model='gpt-4-turbo',
                        messages=[{"role": "user", "content": full_prompt}],
                        # api_key=os.getenv(args.key),
                        # max_tokens=512,
                        temperature=0.0,
                        top_p=1,
                        n=1,
                    )

                    got_result = True
                except Exception as e:
                    print(str(e))
                    sleep(3)
        else:
            # self-consistency decoding
            got_result = False
            while not got_result:
                try:
                    result = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo',
                        messages=[{"role": "user", "content": full_prompt}],
                        # api_key=os.getenv(args.key),
                        # max_tokens=512,
                        temperature=0.5,
                        top_p=1,
                        n=30,
                    )
                    got_result = True
                except Exception as e:
                    print(str(e))
                    sleep(3)

        result_counter = Counter()
        # if self_consistency==1:
        codes = []
        for i in result['choices']:
            res = i['message']['content'].replace('```', '')
            res = res.replace('python\n', '')
            print(res)
            codes.append(res)
        # else:
        #    codes = result['choices'][0]['message']['content']
        # codes = result.split('\n')
        print(codes)

        # codes = parse_api_result(result)
        # handle the s&p500 case
        codes = [code.replace('&', '_') for code in codes]
        dict_pre_code = {}
        for r in codes:
            r = example['generated'][0] + '\n' + r
            print(r)
            ans = safe_execute(r)
            ans = floatify_ans(ans)
            if ans is not None:
                result_counter.update([ans])
                dict_pre_code[ans] = r
        common_code = ''
        if len(result_counter) > 0:
            prediction = result_counter.most_common(1)[0][0]
            common_code = dict_pre_code[prediction]
        else:
            prediction = None

        print(prediction)

        # Further Process according to FinQA dataset
        if prediction is not None:
            if type(prediction) == bool:
                if prediction:
                    prediction = 'yes'
                else:
                    prediction = 'no'
            elif type(prediction) == list:
                prediction = prediction[0]
            else:
                assert type(prediction) in [float, int], prediction

        example.update({'generated': [common_code], 'executed': prediction})

        if prediction is None:
            wrong += 1
        elif finqa_equal(prediction, example['answer'], False):
            correct += 1
            writer_c.write(json.dumps(example) + '\n')
        else:
            wrong += 1

        print('accuracy: ', correct / (correct + wrong))

        writer.write(json.dumps(example) + '\n')

    print()
    print('accuracy: ', correct / (correct + wrong))
    writer.close()
