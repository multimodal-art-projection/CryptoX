# data_construct/utils.py

# python自带的库
import argparse
import logging
import os
import sys

# 常用的开源库
import pandas as pd
import numpy as np

# 项目的库
from .prompt_create import prompt_mmlu, prompt_bbh, prompt_math, prompt_mbpp, prompt_needle

# 配置 logging，设置日志级别和输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_directory(path, mode):
    """
    检查输入、输出目录是否存在
    """
    if not os.path.isdir(path) and mode == "o":
        os.makedirs(path)
        return True
    elif not os.path.isdir(path) and mode == "i":
        return False
    else:
        return True
    

def check_type(p_type):
    """
    检查type类型是否合理
    """
    if p_type in ["simple", "base", "decode", "multi-rounds"]:
        logging.info(f"The prompt type is: {p_type}")
    else:
        logging.error(f"please input right prompt type")
        sys.exit(1)


def check_encode_type(encode_type):
    """
    检查encode_type类型是否合理
    """
    if encode_type in ["nums", "percentages"]:
        logging.info(f"The encode type is: {encode_type}")
    else:
        logging.error(f"please input right encode type")
        sys.exit(1)


def check_nums(nums):
    """
    检查nums类型是否合理
    """
    n_list = nums.split(" ")
    try:
        n_list = [int(num) for num in n_list]
        logging.info(f"The encoding scale employed: {n_list}")
    except:
        logging.error(f"please input right nums")
        sys.exit(1)
    return n_list
    

def check_percentages(percentages):
    """
    检查percentages是否合理
    """
    p_list = percentages.split(" ")
    try: 
        start_p = float(p_list[0])
        end_p = float(p_list[2])
        step = int(p_list[1])
        float_list = np.linspace(start_p, end_p, step).tolist()
        float_list = [round(num, 2) for num in float_list]
        logging.info(f"The encoding scale employed: {float_list}")
    except:
        logging.error(f"please input right percentages")
        sys.exit(1)
    return float_list
    

def check_rule(rule):
    """
    检查编码规则是否存在
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_json(os.path.join(current_directory, "rules.json"))
    if rule not in list(df.columns):
        return False
    logging.info(f"Rules that exist: {list(df.columns)}")
    return True


def check_domain(domain, p_type):
    """
    检查特定子集是否存在
    """
    if domain not in ["all", "math", "bbh", "mbpp", "mmlu", "needle", "subset"]:
        logging.error(f"A domain that doesn't exist: {domain}")
        sys.exit(1)
    elif domain == "all":
        if p_type == "decode" or p_type == "base":
            domain = ["math", "bbh", "mbpp", "mmlu"]
            logging.info(f"Domain: {domain}")
            return domain
        elif p_type == "multi-rounds":
            domain = ["mmlu"]
            logging.info(f"Domain: mmlu")
            return domain
        else:
            domain = ["math", "bbh", "mbpp", "mmlu", "needle"]
            logging.info(f"Domain: {domain}")
            return domain
    else:
        if p_type == "decode" and domain not in ["math", "bbh", "mbpp", "mmlu"]:
            logging.error(f"Domain is not allowed in decode mode: {domain}")
            sys.exit(1)
        elif p_type == "multi-rounds" and domain != "mmlu":
            logging.error(f"Domain is not allowed in multi-rounds mode: {domain}")
            sys.exit(1)
        elif p_type == "base" and domain not in ["math", "bbh", "mbpp", "mmlu"]:
            logging.error(f"Domain is not allowed in base mode: {domain}")
            sys.exit(1)
        logging.info(f"Domain: {domain}")
        return [str(domain)]


def get_rule(rule):
    """
    获取所需的编码规则
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_json(os.path.join(current_directory, "rules.json"))
    for column in df.columns:
        if rule == column:
            rule_item = df[column].to_dict()
            return rule_item
    return False


def get_prompt(dataset_type, code_if, dic, p_type):
    """
    根据获取到的dataset_type构造相应的用于主实验的prompt
    """
    if dataset_type == "math_500":
        if p_type == "simple":
            return prompt_math.simple_common(example_num=0, code_if=code_if, dic=dic)
        elif p_type == "decode":
            return prompt_math.decode_common(example_num=5, dic=dic)
        elif p_type == "base":
            return prompt_math.simple_common(example_num=5, code_if=code_if, dic=dic)
    elif dataset_type == "bbh_405":
        if p_type == "simple":
            return prompt_bbh.simple_common(example_num=3, code_if=code_if, dic=dic)
        elif p_type == "decode":
            return prompt_bbh.decode_common(example_num=5, dic=dic)
        elif p_type == "base":
            return prompt_bbh.simple_common(example_num=5, code_if=code_if, dic=dic)
    elif dataset_type == "mbpp_427":
        if p_type == "simple":
            return prompt_mbpp.simple_common(example_num=0, code_if=code_if, dic=dic)
        elif p_type == "decode":
            return prompt_mbpp.decode_common(example_num=5, dic=dic)
        elif p_type == "base":
            return prompt_mbpp.simple_common(example_num=5, code_if=code_if, dic=dic)
    elif dataset_type == "mmlu_dev_285_hop_1":
        if p_type == "simple":
            return prompt_mmlu.simple_common(example_num=0, code_if=code_if, dic=dic, hop_num=1)
        elif p_type == "decode":
            return prompt_mmlu.decode_common(example_num=5, dic=dic)
        elif p_type == "base":
            return prompt_mmlu.simple_common(example_num=5, code_if=code_if, dic=dic, hop_num=1)
        elif p_type == "multi-rounds":
            return prompt_mmlu.cr_simple(dic=dic, hop_num=1)
    elif dataset_type == "mmlu_dev_285_hop_2":
        if p_type == "simple":
            return prompt_mmlu.simple_common(example_num=0, code_if=code_if, dic=dic, hop_num=2)
        elif p_type == "decode":
            return prompt_mmlu.decode_common(example_num=5, dic=dic)
        elif p_type == "base":
            return prompt_mmlu.simple_common(example_num=5, code_if=code_if, dic=dic, hop_num=2)
        elif p_type == "multi-rounds":
            return prompt_mmlu.cr_simple(dic=dic, hop_num=2)
    elif dataset_type == "mmlu_dev_285_hop_3":
        if p_type == "simple":
            return prompt_mmlu.simple_common(example_num=0, code_if=code_if, dic=dic, hop_num=3)
        elif p_type == "decode":
            return prompt_mmlu.decode_common(example_num=5, dic=dic)
        elif p_type == "base":
            return prompt_mmlu.simple_common(example_num=5, code_if=code_if, dic=dic, hop_num=3)
        elif p_type == "multi-rounds":
            return prompt_mmlu.cr_simple(dic=dic, hop_num=3)
    elif dataset_type == "needle_100":
        return prompt_needle.needle_0shot_template_new
    else:
        logging.error(f"Unknown dataset_type")
        sys.exit(1)
    

def parse_init():
    """
    定义并解析data_encode代码的命令行参数，配置日志记录，并检查输入的编码的数据文件目录和输出的目录是否存在。
    """
    parser = argparse.ArgumentParser(description="Data creation utility")

    # 添加命令行参数
    parser.add_argument('-r', '--rule', type=str, required=True, default="morse_base", help='对数据编码的模式要求,包含morse_base, emoji_morse, emoji_shuffle')
    parser.add_argument('-i', '--input', type=str, default="data_construct/ori_data", help='要进行编码的数据文件的目录地址')
    parser.add_argument('-o', '--output', type=str, default="data_construct/crypto_data", help='编码的数据文件输出的地址')
    parser.add_argument('-p', '--percentages', type=str, default="0 3 1", help='编码的比例，用空格键隔开，第一个是开始的比例，第二个是需要切分成多少个，第三个是结束的比例')
    parser.add_argument('-d', '--domain', type=str, default="all", help='选择的数据类型，从已有的种类中挑选，包含math、bbh、mbpp、mmlu、needle、subset')
    parser.add_argument('-t', '--type', type=str, default="simple", help='选择的prompt的类型，支持simple、base、decode、multi-rounds')
    parser.add_argument('-n', '--nums', type=str, default="0 3 5 10", help='编码的words个数，用list表示，一旦设置')
    parser.add_argument('--encode_type', type=str, default="percentages", help='选择编码单词的类型，是按照个数来编码还是按照比例来编码，有两种编码方式：percentages和nums')

    # 解析命令行参数
    args = parser.parse_args()

    if check_rule(args.rule):
        logging.info(f'Using rule: {args.rule}')
    else:
        logging.error(f"Rule is not exists in rules.json: {args.rule}")
        sys.exit(1)

    if check_directory(args.input, "i"):
        logging.info(f"Input directory: {os.path.abspath(args.input)}")
    else:
        logging.error(f"Input directory is not exists: {os.path.abspath(args.input)}")
        sys.exit(1)

    args.output = os.path.join(args.output, args.rule)
    if check_directory(args.output, "o"):
        logging.info(f"Output directory: {os.path.abspath(args.output)}")

    check_type(args.type)
    check_encode_type(args.encode_type)
    
    if args.encode_type == "nums":
        args.nums = check_nums(args.nums)
    elif args.encode_type == "percentages":
        args.percentages = check_percentages(args.percentages)
    args.domain = check_domain(args.domain, args.type)

    return args