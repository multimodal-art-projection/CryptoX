# python自带的库
import argparse
import json
import math
import csv
from tqdm import tqdm
import os
# 常用的开源库
import torch
# 项目的库
from llamawrapper import  LlamaHelper
import neuron_drawer

def relu(x: float) -> float:
    return max(0.0, x)

def calculate_max_activation(activation_record):
    """
    输出最大的激活值
    """
    return max(relu(x) for x in activation_record['activations'])
     
def normalize_activations(activation_record: list[float], max_activation: float) -> list[int]:
    """
    将原始activation映射到[0,10]
    """
    if max_activation <= 0:
        return [0 for x in activation_record]
    # 使用relu进行归一化并映射到0~10的整数区间
    return [min(10, math.floor(10 * relu(x) / max_activation)) for x in activation_record]

#定义并解析传入logit_lens的命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="Qwen2.5-7B-Instruct",help="测试的模型名字，需要和设置的一致")
parser.add_argument("--model_path", "-p", type=str, default="Qwen2.5-7B-Instruct",help="测试的模型路径，需要和设置的一致")
parser.add_argument("--outputpath", "-o", type=str, default='base_morse',help="输出结果的路径，需要和设置的一致")
args = parser.parse_args()
model_size = args.model
custom_model = args.model_path

# 初始化 LlamaHelper
llama = LlamaHelper(dir=custom_model, load_in_4bit=True, device_map='auto')
tokenizer = llama.tokenizer

# 准备输入提示
question={"prompt_id":"crypto_228","ori_question":"Before Tolstoy's Christian conversion, what was his perspective on the meaning of life?","choices_a":"optimist","choices_b":"satisfied","choices_c":"nominally religious","choices_d":"pessimist","ori_answer":"D","answer":"D","crypto_word_count":10,"question":"-...|.|..-.|---|.-.|. Tolstoy's -.-.|....|.-.|..|...|-|..|.-|-. conversion, .--|....|.-|- .--|.-|... ....|..|... .--.|.|.-.|...|.--.|.|-.-.|-|..|...-|. ---|-. -|....|. --|.|.-|-.|..|-.|--. ---|..-. life?","crypto_word":["-...|.|..-.|---|.-.|.","-.-.|....|.-.|..|...|-|..|.-|-.",".--|....|.-|-",".--|.-|...","....|..|...",".--.|.|.-.|...|.--.|.|-.-.|-|..|...-|.","---|-.","-|....|.","--|.|.-|-.|..|-.|--.","---|..-."],"sample_word":["Before","Christian","what","was","his","perspective","on","the","meaning","of"],"cate":"philosophy","tag":"encode_pct_1,philosophy,单轮","black_white_box_type":"黑盒","rule":"Some of the words in the question are encoded in Morse code, the dictionary of Morse code is:\n    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',\n    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',\n    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',\n    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',\n    'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---',\n    '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',\n    '8': '---..', '9': '----.'\n    You can translate and then answer the questions."}
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
logits = llama.get_logits(prompt)
header = ["layer_idx", "neuron_idx", "vocab", "encoded","vocab_activation_avg","encoded_activation_avg"]
final_header = ["layer_idx",  "vocab", "encoded"]
os.makedirs(f'result/neurons_output', exist_ok=True)
final_output_filename = f"result/neurons_output/{model_size}_final_output.csv"
with open(final_output_filename, "w", newline="", encoding="utf-8") as finalfile:
    final_writer = csv.writer(finalfile)
    final_writer.writerow(final_header)
    for layer_idx, layer_wrapper in tqdm(enumerate(llama.model.model.layers),
                                        total=len(llama.model.model.layers),
                                        desc="Processing Layers"):
        mlp_wrapper = layer_wrapper.block.mlp
        if mlp_wrapper.post_activation is not None:
            post_activation = mlp_wrapper.post_activation
            seq_length = post_activation.shape[1]
            hidden_dim = post_activation.shape[2]
            layer_count1=0
            layer_count2=0
            # 为本层创建独立的 CSV 文件
            os.makedirs(f'result/{model_size}', exist_ok=True)
            layer_output_filename = f"result/{model_size}/neuron_result_{layer_idx}.csv"
            with open(layer_output_filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                # 写入表头
                writer.writerow(header)
                # 遍历神经元
                for neuron_idx in range(hidden_dim):
                    # 收集该神经元所有token的激活值
                    neuron_activations = []
                    for token_idx in range(seq_length):
                        neuron_value = float(post_activation[0, token_idx, neuron_idx].item())
                        neuron_activations.append(neuron_value)
                    
                    # 计算最大激活值并归一化
                    max_act = calculate_max_activation({"activations": neuron_activations})
                    normalized_values = normalize_activations(neuron_activations, max_act)
                    count1=0
                    count2=0
                    sum1=0
                    sum2=0
                    # 写入归一化后的数据
                    for token_idx, norm_val in enumerate(normalized_values):
                        token_word = tokenized_word_list[token_idx]
                        if token_idx>=222 and token_idx<=435:
                            sum1+=norm_val
                            if norm_val >=7:
                                count1+=1
                        if token_idx>=442 and token_idx<=542:
                            sum2+=norm_val
                            if norm_val >=7:
                                count2+=1
                    output_rows=[layer_idx, neuron_idx, count1, count2,round(sum1/214,2),round(sum2/101,2)]
                    if count1 !=0:
                            layer_count1+=1
                    if count2 !=0:
                            layer_count2+=1
                    if count1>=10 or count2>=10:
                        writer.writerow(output_rows)
        
            layer_rows=[layer_idx,  layer_count1, layer_count2]
            final_writer.writerow(layer_rows)
        else:
            print(f"第 {layer_idx + 1} 层没有捕获到 MLP 激活值。")

print("所有层的 MLP 激活值已分别保存为单独的 CSV 文件。")
neuron_drawer.draw(model_size)