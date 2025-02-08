# data_construct/special_encode/needle_data_encode.py
# python自带的库
import glob
from typing import Optional
import os
from datetime import datetime, timezone
import argparse
import logging
import sys
import math

# 常用的开源库
from jsonargparse import CLI
import tiktoken
import pandas as pd
from tqdm import tqdm

# 项目的库
from ..utils import check_directory, check_rule, get_rule, check_percentages
from ..encode_lib import sample_idx, morse_code
from ..prompt_create import prompt_needle


# 配置 logging，设置日志级别和输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Args():
    """
    处理数据时的所需参数
    """
    encode_type: str = "normal"
    needle: Optional[str] = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    haystack_dir: Optional[str] = "PaulGrahamEssays"
    retrieval_question: Optional[str] = "What is the best thing to do in San Francisco?"
    context_lengths_min: Optional[int] = 500
    context_lengths_max: Optional[int] = 130000
    context_lengths_num_intervals: Optional[int] = 35
    context_lengths: Optional[list[int]] = None
    document_depth_percent_min: Optional[int] = 0
    document_depth_percent_max: Optional[int] = 100
    document_depth_percent_intervals: Optional[int] = 35
    document_depth_percents: Optional[list[int]] = None
    save_results: Optional[bool] = True
    final_context_length_buffer: Optional[int] = 200
    print_ongoing_status: Optional[bool] = True
    # Multi-needle parameters
    multi_needle: Optional[bool] = False
    needles: list[str] = [
        " Figs are one of the secret ingredients needed to build the perfect pizza. ", 
        " Prosciutto is one of the secret ingredients needed to build the perfect pizza. ", 
        " Goat cheese is one of the secret ingredients needed to build the perfect pizza. "
    ]
    answer: str = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."


class tokenizer_initial():

    """
    编码器初始化
    """
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("o200k_base")


    def encode_text_to_tokens(self, context):
        """
        将文本转为tokens
        """
        tokens = self.tokenizer.encode(context)
        return tokens
    

    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        将tokens转为文本
        """
        new_context = self.tokenizer.decode(tokens[:context_length])
        return new_context
    

    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        """
        根据给定的上下文和检索问题生成用于查询模型的结构化提示。
        """
        return [{
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "user",
                "content": context
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the long text or repeat your findings."
            }]


class DataConstruct():


    def __init__(self, args):
        """
        初始化所需参数
        """
        self.tokenizer = tokenizer_initial()
        self.haystack_dir = args.haystack_dir
        self.context_lengths = args.context_lengths
        self.final_context_length_buffer = args.final_context_length_buffer
        self.needles = args.needles
        self.retrieval_question = args.retrieval_question
        self.save_results = args.save_results
        self.needle = args.needle
        self.answer = args.answer
        self.rule = args.rule
        self.percentages = args.percentages
        self.encode_percentages_or_nums = args.encode_percentages_or_nums

    
    def get_context_length_in_tokens(self, context):
        """
        获取context的tokens长度
        """
        return len(self.tokenizer.encode_text_to_tokens(context))
    

    def read_context_files(self):
        """
        获取生成needle文本的源文本数据
        """
        context = ""
        max_context_length = max(self.context_lengths)
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory
        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
                with open(file, 'r', encoding='utf-8') as f:
                    context += f.read()
        return context
    

    def encode_and_trim(self, context, context_length):
        """
        将上下文编码为标记，并修剪为指定长度。
        """
        tokens = self.tokenizer.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.tokenizer.decode_tokens(tokens, context_length)
        return context


    def generate_context(self, context_length, depth_percent):
        """
        生成needle数据集所需的文本，包含needle的编码和needle的插入
        """
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = self.insert_needles(context, depth_percent, context_length)
        return context
    

    def insert_needles(self, context, depth_percent, context_length):
        """
        插入多个needles在文本中
        """
        tokens_context = self.tokenizer.encode_text_to_tokens(context)
        context_length -= self.final_context_length_buffer

        # Calculate the total length of all needles in tokens
        total_needles_length = sum(len(self.tokenizer.encode_text_to_tokens(needle)) for needle in self.needles)

        # Ensure context length accounts for needles
        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[:context_length - total_needles_length]
        
        # To evenly distribute the needles, we calculate the intervals they need to be inserted.
        depth_percent_interval = (100 - depth_percent) / len(self.needles)
        
        # Reset the insertion percentages list for the current context
        self.insertion_percentages = []

        # Insert needles at calculated points
        for needle in self.needles:

            tokens_needle = self.tokenizer.encode_text_to_tokens(needle)

            if depth_percent == 100:
                # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
                tokens_context = tokens_context + tokens_needle
            else:
                # Go get the position (in terms of tokens) to insert your needle
                insertion_point = int(len(tokens_context) * (depth_percent / 100))

                # tokens_new_context represents the tokens before the needle
                tokens_new_context = tokens_context[:insertion_point]

                # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
                period_tokens = self.tokenizer.encode_text_to_tokens('.')
                
                # Then we iteration backwards until we find the first period
                while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                    insertion_point -= 1
                    tokens_new_context = tokens_context[:insertion_point]
                    
                # Insert the needle into the context at the found position
                tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]

                # Log 
                insertion_percentage = (insertion_point / len(tokens_context)) * 100
                self.insertion_percentages.append(insertion_percentage)
                # print(f"Inserted '{needle}' at {insertion_percentage:.2f}% of the context, total length now: {len(tokens_context)} tokens")
                
                # Adjust depth for next needle
                depth_percent += depth_percent_interval  

        new_context = self.tokenizer.decode_tokens(tokens_context)
        return new_context
    

    def encode_text(self, text, percentage):
        """
        编码单个text的主方法
        """
        crypto_idx = sample_idx(text, self.rule["rule"], percentage, self.encode_percentages_or_nums)
        crypto_word = []
        sample_word = []
        words = text.split(' ')
        for j in range(len(words)):
            if j in crypto_idx:
                sample_word.append(words[j])
                words[j] = morse_code(words[j], self.rule["rule"])
                crypto_word.append(words[j])
        crypto_text =' '.join(words)
        crypto_word_count = len(crypto_word)
        return crypto_text, crypto_word_count, crypto_word, sample_word
    

    def encode_texts(self, texts, percentage):
        """
        编码多个texts的主方法
        """
        crypto_texts = []
        crypto_word_counts = []
        crypto_words = []
        sample_words = []
        for text in texts:
            crypto_idx = sample_idx(text, self.rule["rule"], percentage, self.encode_percentages_or_nums)
            crypto_word = []
            sample_word = []
            words = text.split(' ')
            for j in range(len(words)):
                if j in crypto_idx:
                    sample_word.append(words[j])
                    words[j] = morse_code(words[j], self.rule["rule"])
                    crypto_word.append(words[j])
            crypto_texts.append(' '.join(words))
            crypto_word_counts.append(len(crypto_word))
            crypto_words.append(crypto_word)
            sample_words.append(sample_word)
        return crypto_texts, crypto_word_counts, crypto_words, sample_words


    def run_sample(self, context_length, depth_percent, code_type, percentage, idx):
        """
        构造单个jsonl文件数据集
        """
        if math.isclose(percentage, 0, rel_tol=1e-9):
            code_type = "normal"
            crypto_word_count = 0
            crypto_word = ""
            sample_word = ""
            ori_question = self.retrieval_question
            ori_needles = self.needles
        if code_type == "question":
            ori_question = self.retrieval_question
            ori_needles = self.needles
            self.retrieval_question, crypto_word_count, crypto_word, sample_word = self.encode_text(self.retrieval_question, percentage)
            ori_needles = self.needles
        elif code_type == "needles":
            ori_question = self.retrieval_question
            ori_needles = self.needles
            self.needles, crypto_word_count, crypto_word, sample_word = self.encode_texts(self.needles, percentage)
        context = self.generate_context(context_length, depth_percent)
        prompt = self.tokenizer.generate_prompt(context, self.retrieval_question)
        result = {
            'context': context.strip(),
            'system_prompt': prompt[0]["content"],
            'user1_prompt': prompt[1]["content"],
            'user2_prompt': prompt[2]["content"],
            'question': self.retrieval_question,
            'ori_question': ori_question,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'needles' : self.needles,
            'ori_needles' : ori_needles,
            'percentage': percentage,
            'code_type': code_type,
            'crypto_word_count': crypto_word_count,
            'crypto_word': crypto_word,
            'answer': self.answer,
            'crypto_id': f'crypto_{str(idx)}',
            'rule': "\n" + str(self.rule["prompt_rule"]).replace("question ", "long text "),
            "dataset_type": "needle_100"
            }
        if math.isclose(percentage, 0, rel_tol=1e-9):
            result["rule"] = ""
        prompt_template = prompt_needle.needle_0shot_template_new.format(**result)
        result["prompt"] = [{"content": prompt_template, "role": "user"}]
        self.needles = ori_needles
        self.retrieval_question = ori_question
        
        return result
    

    def run(self, context_lengths, document_depth_percents, code_type, output_dir):
        """
        构造多个jsonl文件数据集，通过调用run_sample实现
        """
        logging.info(f"Encoding data in needle with context_lengths: {context_lengths}; document_depth_percents: {document_depth_percents}......")
        for percentage in self.percentages:
            logging.info(f"Encoding data in needle with percentage: {percentage}")
            res = []
            idx = 0
            for context_length in tqdm(context_lengths, total=len(context_lengths), desc="Encoding......"):
                for document_depth_percent in document_depth_percents:
                    result = self.run_sample(context_length, document_depth_percent, code_type, percentage, idx)
                    res.append(result)
            df = pd.DataFrame(res)
            fout = f'{output_dir}/needle_{percentage}_{context_lengths[-1] // 1000}_{code_type}.jsonl'
            if percentage == 0:
                fout = f"{output_dir}/needle_{percentage}_{context_lengths[-1] // 1000}.jsonl"
            df.to_json(fout, orient='records', lines=True, force_ascii=False)
        

def cmd_parse_init():
    """
    从cmd中获取命令行参数
    """
    parser = argparse.ArgumentParser(description="Needle Data creation utility")

    # 添加命令行参数
    parser.add_argument('-r', '--rule', type=str, required=True, default="morse_base", help='对数据编码的模式要求,包含morse_base,emoji_morse,emoji_shuffle')
    parser.add_argument('-o', '--output', type=str, default="data_construct/crypto_data", help='编码的数据文件输出的地址')
    parser.add_argument('-p', '--percentages', type=str, default="0 3 1", help='编码的比例，用空格键隔开，第一个是开始的比例，第二个是需要切分成多少个，第三个是结束的比例')
    parser.add_argument('-ddp', '--document_depth_percents', type=str, default="0 12 23 34 45 56 67 78 89 100", help='needle在文本中的深度，用空格键隔开')
    parser.add_argument('-cl', '--context_lengths', type=str, default="1000 4000 7000 10000 13000 16000 19000 22000 26000 30000", help='文本的长度，用空格键隔开')
    
    args = parser.parse_args()

    # 解析命令行参数

    if check_rule(args.rule):
        logging.info(f'Using rule: {args.rule}')
    else:
        logging.error(f"Rule is not exists in rules.json: {args.rule}")
        sys.exit(1)
    
    args.output = os.path.join(args.output, args.rule)
    if check_directory(args.output, "o"):
        logging.info(f"Output directory: {os.path.abspath(args.output)}")
    
    args.percentages = check_percentages(args.percentages)

    return args


def encode_needle(percentages, rule, output, encode_type):
    """
    外部调用代码以完成对needle数据集的编码，只不过此时设置的文本长度和深度固定
    ddp: "0 12 23 34 45 56 67 78 89 100"
    cl: "1000 4000 7000 10000 13000 16000 19000 22000 26000 30000"
    """
    args = Args()
    args.context_lengths = [int(context_length) for context_length in "1000 4000 7000 10000 13000 16000 19000 22000 26000 30000".split(" ")]
    args.document_depth_percents = [int(document_depth_percent) for document_depth_percent in "0 12 23 34 45 56 67 78 89 100".split(" ")]

    args.percentages = percentages
    args.rule = rule
    args.encode_percentages_or_nums = encode_type

    args.multi_needle = True
    if args.multi_needle:
        args.retrieval_question = "What are the secret ingredients needed to build the perfect pizza?"
        args.answer = "figs, prosciutto and goat cheese."
    
    dc = DataConstruct(args)
    dc.run(args.context_lengths, args.document_depth_percents, "needles", output)


def main():
    """
    主代码块，完成needle数据集的编码
    """
    cmd_args = cmd_parse_init()
    args = Args()

    try:
        context_lengths = [int(context_length) for context_length in cmd_args.context_lengths.split(" ")]
        args.context_lengths = context_lengths
    except:
        logging.ERROR("Please input right context_lengths")

    try:
        document_depth_percents = [int(document_depth_percent) for document_depth_percent in cmd_args.document_depth_percents.split(" ")]
        args.document_depth_percents = document_depth_percents
    except:
        logging.ERROR("Please input right document_depth_percents")


    args.percentages = cmd_args.percentages
    args.rule = get_rule(cmd_args.rule)

    args.multi_needle = True
    if args.multi_needle:
        args.retrieval_question = "What are the secret ingredients needed to build the perfect pizza?"
        args.answer = "figs, prosciutto and goat cheese."


    dc = DataConstruct(args)
    dc.run(args.context_lengths, args.document_depth_percents, "needles", os.path.abspath(cmd_args.output))


if __name__ == "__main__":
    main()