# data_construct/data_encode.py
# python自带的库
import os
import logging
import math

# 常用的开源库
import pandas as pd
from tqdm import tqdm

# 项目的库
from .utils import parse_init, get_rule, get_prompt
from .encode_lib import sample_idx, morse_code
from .special_encode.needle_data_encode import encode_needle


# 配置 logging，设置日志级别和输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def one_file_encode(file, df, math_rule, rule, percentage, output_dir_path, p_type, encode_type):
    """
    对一个数据集按照percentage进行编码
    """
    data = []
    for idx, item in df.iterrows():
        item = item.to_dict()
        question = item["question"]
        if item.get("dataset_type", "no_type") == "math_500":
            rule_chosen = math_rule
        else:
            rule_chosen = rule
        crypto_idx = sample_idx(question, rule_chosen["rule"], percentage, encode_type)
        crypto_word = []
        sample_word = []

        words = question.split(' ')
        if not math.isclose(percentage, 0, rel_tol=1e-9):
            for j in range(len(words)):
                if j in crypto_idx:
                    sample_word.append(words[j])
                    words[j] = morse_code(words[j], rule_chosen["rule"])
                    crypto_word.append(words[j])
        
        crypto_question =' '.join(words)
        crypto_word_count = len(crypto_word)

        item["question"] = crypto_question
        item["ori_question"] = question
        item["rule"] = rule_chosen["prompt_rule"]
        item["crypto_word_count"] = crypto_word_count
        item['crypto_word'] = crypto_word
        item['sample_word'] = sample_word
        item["crypto_id"] = f'crypto_{str(idx)}'

        if p_type != "multi-rounds":
            if math.isclose(percentage, 0, rel_tol=1e-9):
                prompt = get_prompt(item.get("dataset_type", "no_type"), code_if=False, dic=rule_chosen["rule"], p_type=p_type)
            else:
                prompt = get_prompt(item.get("dataset_type", "no_type"), code_if=True, dic=rule_chosen["rule"], p_type=p_type)
            prompt = prompt.format(**item)
            if p_type == "decode":
                prompt = prompt.replace("You can translate and then answer the questions. ", "")
            item["prompt"] = [{"content": prompt, "role": "user"}]

        else:
            if math.isclose(percentage, 0, rel_tol=1e-9):
                return
            else:
                prompt = get_prompt(item.get("dataset_type", "no_type"), code_if=True, dic=rule_chosen["rule"], p_type=p_type)
                prompt[0] = prompt[0].format(**item).replace("You can translate and then answer the questions. ", "")
                item["prompt"] = [{"content": prompt[0], "role": "user"}, {"content": prompt[1].format(**item), "role": "user"}]

        data.append(item)
    df = pd.DataFrame(data)
    fout = os.path.join(output_dir_path, f"{file.replace('.jsonl', '')}_percentage_{percentage}.jsonl")
    df.to_json(fout, orient='records', lines=True, force_ascii=False)
    

def files_encode(math_rule, rule, input_dir_path, percentages, output_dir_path, domain, p_type, encode_type):
    """
    对多个数据集进行编码，调用one_file_encode以完成编码
    """
    logging.info(f"Encoding data in directory: {input_dir_path}......")
    files = os.listdir(input_dir_path)
    mmlu_num = 0
    for file in files:
        if not any(element in file for element in domain):
            continue
        if "mmlu" in file:
            mmlu_num += 1
            if mmlu_num > 1 and p_type == "decode":
                continue
        file_path = os.path.join(input_dir_path, file)
        df = pd.read_json(file_path, lines=True)
        logging.info(f"Encoding data in file: {file}......")
        for percentage in percentages:
            logging.info(f"Encoding data in file: {file} with {encode_type[:-1]}: {percentage}")
            one_file_encode(file, df, math_rule, rule, percentage, output_dir_path, p_type, encode_type)

    # 对needle数据集进行编码
    if "needle" in domain:
        encode_needle(percentages, rule, output_dir_path, encode_type)


def main():
    """
    主代码块，进行数据集的编码操作
    """
    args = parse_init()

    if args.rule in ["morse_base", "emoji_morse", "emoji_shuffle"]:
        rule = get_rule(args.rule)
        math_rule = get_rule(args.rule+"_math")
    else:
        rule = get_rule(args.rule)
        math_rule = get_rule(args.rule)
    input_dir_path = os.path.abspath(args.input)
    output_dir_path = os.path.abspath(args.output)
    if args.encode_type == "percentages":
        perentages = args.percentages
    else:
        perentages = args.nums
    domain = args.domain
    p_type = args.type
    encode_type = args.encode_type

    files_encode(math_rule, rule, input_dir_path, perentages, output_dir_path, domain, p_type, encode_type)
    logging.info("Encoding is complete")


if __name__ == "__main__":
    main()