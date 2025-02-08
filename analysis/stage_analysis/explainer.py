# python自带的库
import glob
import os
import argparse
import json
# 项目的库
from example import ORIGINAL_EXAMPLES
from utils import calculate_max_activation,normalize_activations,calculate_max_logit
from response import get_response
from prompt_template import summarizing_prompt

processed_example=ORIGINAL_EXAMPLES.copy()
question={"prompt_id":"crypto_228","ori_question":"Before Tolstoy's Christian conversion, what was his perspective on the meaning of life?","choices_a":"optimist","choices_b":"satisfied","choices_c":"nominally religious","choices_d":"pessimist","ori_answer":"D","answer":"D","crypto_word_count":10,"question":"-...|.|..-.|---|.-.|. Tolstoy's -.-.|....|.-.|..|...|-|..|.-|-. conversion, .--|....|.-|- .--|.-|... ....|..|... .--.|.|.-.|...|.--.|.|-.-.|-|..|...-|. ---|-. -|....|. --|.|.-|-.|..|-.|--. ---|..-. life?","crypto_word":["-...|.|..-.|---|.-.|.","-.-.|....|.-.|..|...|-|..|.-|-.",".--|....|.-|-",".--|.-|...","....|..|...",".--.|.|.-.|...|.--.|.|-.-.|-|..|...-|.","---|-.","-|....|.","--|.|.-|-.|..|-.|--.","---|..-."],"sample_word":["Before","Christian","what","was","his","perspective","on","the","meaning","of"],"cate":"philosophy","tag":"encode_pct_1,philosophy,单轮","black_white_box_type":"黑盒","rule":"Some of the words in the question are encoded in Morse code, the dictionary of Morse code is:\n    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',\n    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',\n    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',\n    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',\n    'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---',\n    '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',\n    '8': '---..', '9': '----.'\n    You can translate and then answer the questions."}
input_prompt = """
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

#构建example input prompt
input_prompt=input_prompt.format(**question)
for i in range(len(ORIGINAL_EXAMPLES)):
    max_activation=calculate_max_activation(ORIGINAL_EXAMPLES[i])
    processed_example[i]['activations']=normalize_activations(ORIGINAL_EXAMPLES[i]['activations'],max_activation)
final_example=f"We are studying the functionality of each layer during LLM inference. We have selected the most likely output tokens for each layer's hidden state. Please check the activation of the hidden state's output for the following tokens and summarize in a single sentence what the layer's function is. Don't list examples of words.\n\nThe activation format is token<tab>activation. Activation values range from 0 to 10. A layer finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.We have provided the input prompt for the LLM. You can use this input prompt as a reference to evaluate the functionality of each layer.\n"

#构建stage analysis的prompt
for i in range(len(processed_example)):
    final_example+=f"Layer {i}\n<start>\n"
    example=""
    example_without_zero=""
    for j in range(len(processed_example[i]['tokens'])):
        example+=f"{processed_example[i]['tokens'][j]}\t{processed_example[i]['activations'][j]}\n"
        if processed_example[i]['activations'][j]!=0:
            example_without_zero+=f"{processed_example[i]['tokens'][j]}\t{processed_example[i]['activations'][j]}\n"
    final_example=final_example+example_without_zero+"<end>\n"
    final_example=final_example+f"Explanation of Layer {i} behavior:the main thing this layer does is to {processed_example[i]['explanation']}\n"

#定义并解析传入的命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="Qwen2.5-7B-Instruct",help="测试的模型名字，需要和设置的一致")
args = parser.parse_args()
model_size = args.model
logit_folder_path = f"result/{model_size}/logit"
json_file_list = glob.glob(os.path.join(logit_folder_path, "*.json"))
layer_func_list=[[] for i in range(len(json_file_list[0]))]

#利用已抽取的logit进行分析
for logit_dict_path in json_file_list:
    print(f"Processing: {logit_dict_path}")
    with open(logit_dict_path, 'r') as f:
        logit_dict = json.load(f)
    for i in range(len(logit_dict)):
        response=f"Layer {i}\n<start>\n"
        max_logit=calculate_max_logit(logit_dict[f'layer{i}'][1])
        token_list=logit_dict[f'layer{i}'][0]
        normalized_logit=normalize_activations(logit_dict[f'layer{i}'][1],max_logit)
        for j in range(len(token_list)):
            response+=f"{token_list[j]}\t{normalized_logit[j]}\n"
        prompt=final_example+response+f"<end>\nThe example input prompt is {input_prompt} and you can use the example input prompt to assist you in making your judgment.\nExplanation of Layer {i} behavior:the main thing this layer does is is to"
        response=get_response(prompt)
        final_response=get_response(summarizing_prompt.format(info=response,input=input_prompt))
        layer_func_list[i].append(final_response)
for i in range(len(logit_dict)):
    os.makedirs(f'result/{model_size}/layer_result', exist_ok=True)
    output_file = f"result/{model_size}/layer_result/layer_{i}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, summary in enumerate(layer_func_list[i]):
            f.write(summary + "\n")
    print(f"Layer {i} results have been saved to {output_file}")
prompt="""
We are studying the functions of LLM layers in inference. Currently, we have obtained the functions exhibited by a certain layer in the inference process with different inputs. Now, we need you to summarize the above functional information and summarize the functions of that layer in concise language. If there are multiple functions, you can use ',' to segment them.
Functions:"""
result_folder_path = f"result/{model_size}/layer_result"
result_file_list = glob.glob(os.path.join(result_folder_path, "*.txt"))
i=0
output_file = f"result/{model_size}/final_explanation.txt"

#对分析结果进行提炼总结
with open(output_file, "w", encoding="utf-8") as f_out:
    for result in result_file_list:
        with open(result, 'r') as f:
            content = f.read()
        prompt=prompt+content+"Please give your summary in one sentence in English:"
        f_out.write(f"layer {i} \n")
        i+=1
        f_out.write(f"{get_response(prompt)} \n")
print(f"Stage analysis result have been saved to {output_file}")