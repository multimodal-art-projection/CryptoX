# python自带的库
import json
import math
import time
import os
import argparse
# 常用的开源库
import torch
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
# 项目的库
from llamawrapper import  LlamaHelper

def relu(x: float) -> float:
    return max(0.0, x)

def calculate_max_activation(activation_record):
    """输出最大的Logit"""
    return max(relu(x) for x in activation_record['activations'])
     
def normalize_activations(activation_record: list[float], max_activation: float) -> list[int]:
    """将原始Logit映射到[0,10]"""
    if max_activation <= 0:
        return [0 for x in activation_record]
    # 使用relu进行归一化并映射到0~10的整数区间
    return [min(10, math.floor(10 * relu(x) / max_activation)) for x in activation_record]

#定义并解析传入的命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="Qwen2.5-7B-Instruct",help="测试的模型名字，需要和设置的一致")
parser.add_argument("--model_path", "-p", type=str, default="Qwen2.5-7B-Instruct",help="测试的模型路径，需要和设置的一致")
args = parser.parse_args()
model_size = args.model
custom_model = args.model_path

llama = LlamaHelper(dir=custom_model, load_in_4bit=True, device_map='auto')
tokenizer = llama.tokenizer

#读取数据集
fin="../logit_lens/data/morse/modified_morse_base_mmlu_dev_285_hop_1_percentage_5.jsonl"
df = pd.read_json(fin, lines=True)
df = df.sample(min(5, len(df)), random_state=int(time.time()))
for idx, item in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):  # 添加进度条
    question= item.to_dict()
    prompt = """
    example:The following paragraphs each describe a set of seven objects arranged in a fixed order. The statements are logically consistent within each paragraph. In an antique car show, there are seven vehicles: a limousine, a truck, a minivan, a tractor, a hatchback, a bus, and a convertible. The convertible is the second-oldest. The tractor is the fourth-newest. The minivan is older than the truck. The truck is older than the limousine. The hatchback is the third-oldest. The bus is newer than the limousine.
    A The limousine is the oldest
    B The truck is the oldest
    C The minivan is the oldest
    D The tractor is the oldest
    Answer:C
    Which of the following is a humorous edit of this artist or movie name: 'the third man'?
    A the third men
    B the trird man
    C thed third man
    D the third pan
    Answer:D
    {rule}

    {question}

    A {choices_a}
    B {choices_b}
    C {choices_c}
    D {choices_d}

    Answer:"""
    prompt=prompt.format(**question)
    token_list=tokenizer.encode(prompt)
    tokenized_word_list=[]
    for token in token_list:
        tokenized_word_list.append(tokenizer.decode(token))

    # 获取 logits
    model = llama.model
    unemb = nn.Sequential(llama.model.model.norm, llama.model.lm_head)
    latents = llama.latents_all_layers(prompt)
    latents = latents.to('cuda')
    logits = unemb(latents)
    last = logits[:, -1, :].float().detach().cpu()
    top_k=30
    output_json={}
    value_list,token_list=torch.topk(last, top_k, dim=-1)
    for i in range(len(value_list)):
        token_output=[]
        value_output=[]
        for j in range(top_k):
            token_output.append(tokenizer.decode(token_list[i][j]))
            value_output.append(float(value_list[i][j]))
        output_json[f'layer{i}']=[token_output,value_output]
    os.makedirs(f'result/{model_size}/logit', exist_ok=True)
    with open(f'result/{model_size}/logit/{question["crypto_id"]}logit_top10.json', 'w') as f:
        json.dump(output_json, f)