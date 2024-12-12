import json
from tqdm import tqdm
operations_dict = {
    'add': '+',
    'subtract': '-',
    'multiply': '*',
    'divide': '/',
    'exp': '^',
    'greater': '>',
    'table_sum': 'sum',
    'table_average': 'average',
    'table_max': 'max',
    'table_min': 'min'
}
'''
TODO
把finqa转成可以访问gpt的数据
'''
def program_tokenization(original_program):
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    return program
def convert_arg(argx):
    if '#' in argx:
        argx = argx.replace('#', 'x')
    if 'const' in argx or 'CONST' in argx:
        argx = argx.replace('const_', '')
        argx = argx.replace('CONST_', '')
        if argx[0] == 'm' or argx[0] == 'M':
            argx = '-' + argx[1:]
    return argx
def convert_program(original_program):
    res = ''
    program_list = program_tokenization(original_program)
    steps = int(len(program_list) / 4)
    for i in range(steps):
        op = program_list[i*4].replace('(', '')
        arg1 = program_list[i*4 + 1]
        arg2 = program_list[i*4 + 2]

        op_new = operations_dict[op]
        arg1 = convert_arg(arg1)
        arg2 = convert_arg(arg2)
        if i == steps - 1:
            v = 'ans'
        else:
            v = 'x' + str(i)
        if 'table' in op:
            res += v + ' = ' + op_new + '(' + arg1 + ')'
        else:
            res += v + ' = ' + arg1 + ' ' + op_new + ' ' + arg2
        if i != steps - 1:
            res += '; '
    return res

def convert_finqa_example(finqa, qa):

    text = ""
    table = ""

    program = ""

    #print(finqa)
    question = finqa[qa]['question']
    answer = finqa[qa]['exe_ans']
    id = finqa['id']


    gold = finqa['qa']['gold_inds']

    for key in gold.keys():
        if 'text' in str(key):
            text += gold[str(key)]

    # for t in finqa['pre_text']:
    #     if t != '.':
    #         text += t
    # for t in finqa['post_text']:
    #     if t != '.':
    #         text += t

    for row in finqa['table']:
        for ind, cow in enumerate(row):
            if ind != len(row) - 1:
                if cow != '':
                    table += cow
                else:
                    table += '-'
                table += ' | '
            else:
                if cow != '':
                    table += cow
                else:
                    table += '-'
                table += '\n'

    program = convert_program(finqa[qa]['program'])

    return {"question": question, "text": text, "table": table, "answer": answer, "program": program, "id": id, "gold": gold}


with open('finqa_train.json') as f:
    finqa_dev = json.load(f)
res = []
for example in tqdm(finqa_dev):
    try:
        if 'qa' in example.keys():
            r = convert_finqa_example(example, 'qa')
            res.append(r)
        if 'qa_0' in example.keys():
            r = convert_finqa_example(example, 'qa_0')
            res.append(r)
        if 'qa_1' in example.keys():
            r = convert_finqa_example(example, 'qa_1')
            res.append(r)
    except Exception as e:
        print(example['id'])

with open('finqa_gold_wocode.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)
