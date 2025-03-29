# data_construct/prompt_create/prompt_needle.py
# Prompt for needle
needle_0shot_template = """Please follow the context given above, answer the following question.
The last line of your response should be of the following format: 'Answer: $YOUR_ANSWER' (without quotes) where YOUR_ANSWER is your final answer of the question and it must be a few words.

Some of the words in the context above are encoded in a code, the dictionary of the code is:
{rule}
You can translate the encoded part in the context and then answer the question below.

## Question
{user2_prompt}

Think step by step before answering.

Remember, the last line of your response must be of the following format: 'Answer: $YOUR_ANSWER' (without quotes) where YOUR_ANSWER is your final answer of the question and it must be a few words"""

needle_0shot_template_new = """I'll give you a long text and a question that you need to answer based on what the long text says. The last line of your response should be of the following format: 'Answer: $YOUR_ANSWER' (without quotes) where YOUR_ANSWER is your final answer of the question.
{rule}

### Long text:
{context}

### Question:
{user2_prompt} Think step by step before answering.

Remember, the last line of your response should be of the following format: 'Answer: $YOUR_ANSWER' (without quotes) where YOUR_ANSWER is your final answer of the question."""
