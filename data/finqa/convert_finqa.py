import json
import random
import os

def select_random_json_items(input_path, output_path, n):
    # 读取JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 随机选择n个项目
    selected_items = random.sample(data, n)

    # 写入到输出路径
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_items, f, ensure_ascii=False, indent=4)


def get_all_files_in_folder(folder_path):
    """
    获取指定文件夹中的所有文件

    参数:
    folder_path (str): 文件夹路径

    返回:
    list: 包含文件路径的列表
    """
    files_list = []
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 遍历文件夹中的所有文件和子文件夹
        for root, dirs, files in os.walk(folder_path):
            # 将每个文件添加到文件列表中
            for file in files:
                files_list.append(os.path.join(root, file))
    else:
        print("文件夹不存在")
    return files_list


def read_file_to_dicts(file_name):
    """
    读取文件内容，每一行作为一个字典

    参数:
    file_name (str): 文件名

    返回:
    list: 包含每一行内容的字典列表
    """
    dict_list = []
    try:
        # 打开文件
        with open(file_name, 'r') as file:
            # 逐行读取文件内容
            for line in file:
                # 创建一个字典，将每一行的内容作为字典的值，键为行号（从1开始）
                line_dict = json.loads(str(line))
                dict_list.append(line_dict)
    except FileNotFoundError:
        print(f"文件 '{file_name}' 不存在")

    return dict_list


def save_dict_list_to_json(dict_list, file_name):
    """
    将字典列表保存为JSON文件

    参数:
    dict_list (list): 字典列表
    file_name (str): 要保存的文件名

    返回:
    bool: True表示保存成功，False表示保存失败
    """
    try:
        # 将字典列表转换为JSON格式的字符串
        json_string = json.dumps(dict_list, indent=4)
        # 将JSON字符串写入文件
        with open(file_name, 'w') as file:
            file.write(json_string)
        return True
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")
        return False

def remove_duplicates_by_id(dict_list):
    """
    根据字典中的'id'字段去除列表中的重复值

    参数:
    dict_list (list): 包含字典的列表

    返回:
    list: 去除重复值后的列表
    """
    # 用于存储已经遇到的id值
    seen_ids = set()
    # 列表推导式，遍历dict_list，仅保留id值不在seen_ids中的字典
    unique_dicts = [d for d in dict_list if not (d['id'] in seen_ids or seen_ids.add(d['id']))]
    return unique_dicts

def round_to_n_decimals(number, n):
    if n >= 0:
        factor = 10 ** n
        return round(number * factor) / factor
    else:  # n < 0
        return round(number, n)

def check_ans(answer, exe_ans):
    answer = answer.replace('$ ', '')
    answer = answer.replace('\n', '')
    if type(exe_ans) is str:
        if answer == exe_ans:
            return True
        else:
            return False
    else:
        if '%' in answer:
            l = len(answer)
            answer = answer[0:l - 1]
            exe_ans *= 100


        if '.' in answer:
            num = len(answer.split('.')[-1])
        else:
            num = 0
        exe_ans = round_to_n_decimals(exe_ans, num)
        if num == 0:
            exe_ans = str(int(exe_ans))
        exe_ans = str(exe_ans)

        if answer == exe_ans:
            return True
        else:
            return False

def data_filter(data_list):
    new_data= []
    error_id = []
    for example in data_list:
        code = example['generated'][0]
        code_list = code.split('\n')
        #error_list = [' - ', 'x0', '+', '*', '/' ]
        error_list = ['x0']
        for i in error_list:
            if i in code_list[0]:
                error_id.append(example['id'])
                print(example['id'])
                break
            new_data.append(example)
    print(len(error_id))
    return new_data

def remove_comment(code):
    code_woc = ''
    code_list = code.split('\n')
    for line in code_list:
        new_line = line
        if '#' in line:
            new_line = line.split('#')[0]
        if new_line != '':
            code_woc += new_line + '\n'

    def clean_content(res):
        return (" ".join([token for token in res.split(" ") if token])).strip()

    code_woc = clean_content(code_woc)
    print(code_woc)
    return code_woc

if __name__ == "__main__":
    f_list = get_all_files_in_folder("jsonl")
    example = []
    for f in f_list:
        example_t = read_file_to_dicts(f)
        example += example_t
    print("num:", len(example))
    example_new = []
    id_c = []
    for e in example:
        if e['id'] in id_c:
            continue
        id_c.append(e['id'])
        example_new.append(e)
    print("num:", len(example_new))
    #save_dict_list_to_json(example_new, "finqa_train7.json")

    for e in example_new:
        e['generated'][0] = remove_comment(e['generated'][0])
    save_dict_list_to_json(example_new, "finqa_train_llm.json")