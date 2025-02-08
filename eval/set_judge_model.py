# eval/set_judge_model.py
# python自带的库
import argparse
import logging
import os
import re


# 配置 logging，设置日志级别和输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_init():
    """
    定义并解析set_judge_model代码的命令行参数，配置日志记录
    """
    parser = argparse.ArgumentParser(description="Data creation utility")

    parser.add_argument("-m", "--model", type=str, required=True, default="deepseek-chat", help="用于判断正确与否的judge模型名字")
    parser.add_argument("-a", "--address", type=str, required=True, default="https://api.deepseek.com", help="部署的judge模型的地址")
    parser.add_argument("-k", "--key", type=str, required=True, default="API-KEY", help="API的key")
    
    # 解析命令行参数
    args = parser.parse_args()

    return args


def modify_name(file_path, new_name_list, name_list):
    """
    替换单个文件中的指定变量
    """
    with open(file_path, 'r', encoding="utf-8") as file:
        content = file.read()

    # 使用正则表达式查找 name 的赋值语句并替换
    for i in range(len(name_list)):
        content = re.sub(f'{name_list[i]}\s*=\s*"[^"]+"', f'{name_list[i]} = "{new_name_list[i]}"', content)

    with open(file_path, 'w', encoding="utf-8") as file:
        file.write(content)


def main():
    """
    主代码，替换3个文件中的model_name, openai_api_base和openai_api_key
    """
    args = parse_init()
    name_list = ["model_name", "openai_api_base", "openai_api_key"]
    new_name_list = [args.model, args.address, args.key]
    file_list = ["metric_bbh.py", "metric_needle.py", "metric_math500.py"]
    current_dir = os.path.dirname(os.path.realpath(__file__))
    for file in file_list:
        file_path = os.path.join(current_dir, "metric_lib", file)
        modify_name(file_path, new_name_list, name_list)


if __name__ == "__main__":
    main()
