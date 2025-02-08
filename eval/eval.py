# eval/eval.py
# python自带的库
import asyncio
import os
import logging

# 常用的开源库
import pandas as pd
from tqdm import tqdm

# 项目的库
from .utils import parse_init
from .eval_lib import predict, metric, save_process


# 配置 logging，设置日志级别和输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def eval_file(file_path, output_dir, model_name, address, key, sem):
    """
    评估单个文件
    """
    df = pd.read_json(file_path, lines=True)
    df = df.sample(len(df), random_state=42)
    item_list = []
    for idx, item in tqdm(df.iterrows(), total=len(df), desc="Processing prompts......"):
        item = item.to_dict()
        item_list.append(item)
    
    item_list = await predict(item_list, sem, model_name, address, key)
    item_list = await metric(item_list, sem)

    file_name = os.path.basename(file_path)
    save_process(item_list, output_dir, file_name)
    logging.info(f"Complete the evaluation of the file: {file_name}")



async def eval_files(dir_path, output_dir, model_name, address, key, sem):
    """
    评估单个目录下的多个文件
    """
    files = os.listdir(dir_path)
    for file in files:
        logging.info(f"Evaluating file: {file}")
        file_path = os.path.join(dir_path, file)
        if not os.path.exists(os.path.join(output_dir, file)):
            await eval_file(file_path, output_dir, model_name, address, key, sem)
        else:
            logging.info(f"File {file} has been evaluated, skip it")


async def main():
    """
    主代码块，进行数据的评估，包括调用模型以及对response进行评估
    """
    sem = asyncio.Semaphore(2)  # 将信号量放在这里
    args = parse_init()
    model_name = args.model
    address = args.address
    key = args.key
    if args.recursive:
        await eval_files(args.input, args.output, model_name, address, key, sem)
    else:
        await eval_file(args.input, args.output, model_name, address, key, sem)


if __name__ == "__main__":
    asyncio.run(main())