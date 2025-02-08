# data_construct/encode_lib.py
# python自带的库
import random


def check_good(sssss, rule_dict):
    """
    检测单个单词中的所有字符是否在rule的keys中
    """
    alist = set(rule_dict.keys())
    for char in sssss:
        char = char.upper()
        if char not in alist:
            return False
    return True


def sample_idx(question, rule_dict, percent=0.2, encode_type="percentages"):
    """
    获取需要进行编码的单词的idx
    """
    # 设置随机种子
    random.seed(42)
    question_word_list = question.split(' ')
    question_word_sample = []
    question_word_sample_idx = []
    for idx, word in enumerate(question_word_list):
        is_good = check_good(word, rule_dict)
        if not is_good:
            continue
        elif len(word) <= 1:
            continue
        elif word in question_word_sample:
            continue
        else:
            question_word_sample_idx.append(idx)
            question_word_sample.append(word)
    if encode_type == "percentages":
        question_word_sample_idx = random.sample(question_word_sample_idx, int(percent * len(question_word_sample_idx)))
    elif encode_type == "nums":
        question_word_sample_idx = random.sample(question_word_sample_idx, min(int(percent), len(question_word_sample_idx)))
    return question_word_sample_idx


def morse_code(text, rule_dict):
    """
    对文本text进行编码
    """
    morse_code_list = []
    for char in text:
        char = char.upper()
        morse_code_list.append(rule_dict[char.upper()])
    morse_code = '|'.join(morse_code_list)
    return morse_code.strip()
