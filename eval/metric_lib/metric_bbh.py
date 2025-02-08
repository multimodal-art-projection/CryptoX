# eval/metric_lib/metric_bbh.py
# python自带的库
import re
import asyncio
import logging
import os

# 常用的开源库
from openai import OpenAI

# 项目的库
from .prompt_template_for_metric import judge_prompt_template_bbh

# 设置OpenAI的API密钥和API库以使用vLLM的API服务器
openai_api_key = "API-KEY"
openai_api_base = "https://api.deepseek.com"
model_name = "deepseek-chat"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def normalize_response(response: str) -> str:
    """
    通过删除可能阻止匹配的markdown和LaTeX格式来规范化响应。
    """
    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def judge_easy(answer, response):
    """
    对答案进行简单的匹配
    """
    answer = answer.upper().replace("(", "").replace(")", "").replace(" ", "").replace("$", "").strip()
    response = response.upper().replace("(", "").replace(")", "").replace(" ", "").replace("$", "").strip()
    return answer == response
    

def judge_correctness_bbh(problem, real_answer, generated_answer):
    """
    判断bbh数据集的答案的正确性
    """
    generated_answer = normalize_response(generated_answer)
    pos = generated_answer.lower().rfind("answer")
    if pos == -1:
        return False
    generated_answer = generated_answer[pos:]
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*(.*)"
    match_for_generated_answer = re.findall(ANSWER_PATTERN_MULTICHOICE, generated_answer)
    if match_for_generated_answer:
        if len(real_answer.strip().split(" ")) > 1:
            try:
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": judge_prompt_template_bbh.format(problem=problem, real_answer=real_answer, generated_answer=match_for_generated_answer[-1])},
                    ]
                )
            except:
                return False
            match = re.findall(r"<answer>(.*?)</answer>", normalize_response(chat_response.choices[-1].message.content))
            if match:
                return match[-1] == "correct"
            else:
                logging.error(f"响应中未找到 <answer> 标签。problem:{problem}\nreal_answer:{real_answer}\ngenerated_answer:{generated_answer}")
                return False  # 或根据需求进行处理
        return judge_easy(real_answer, match_for_generated_answer[-1])
    else:
        return False


async def judge_correctness(problem, real_answer, generated_answer):
    """
    异步版本的用于判断正确性的请求函数，使用运行中的事件循环
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, judge_correctness_bbh, problem, real_answer, generated_answer)
    return result


if __name__ == '__main__':
    problem="The binary number $10101001110_{2}$ is equal to what number in base eight?"
    real_answer="2516_8"
    generated_answer="Answer:8 1"
    print(judge_correctness_bbh(problem, real_answer, generated_answer))