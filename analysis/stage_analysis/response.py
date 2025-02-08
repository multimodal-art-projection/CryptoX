# python自带的库
import re
import logging
# 常用的开源库
from openai import OpenAI

#配置api key和api base
openai_api_key = ""
openai_api_base = ""
logging.basicConfig(level=logging.ERROR)
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
def get_response(prompt):
    """
    用于获取api LLM的响应
    """
    chat_response = client.chat.completions.create(
            model="",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
    return chat_response.choices[-1].message.content
