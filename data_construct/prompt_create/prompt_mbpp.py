# data_construct/prompt_create/prompt_mbpp.py
# Python Standard Library
import copy
# Components in prompt
guide_nocode = """You are a expert Python programmer, I'll give you a question.\nYour task is:\n- First, answer the question.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question."""

guide_code = """You are a expert Python programmer, I'll give you a code dictionary and a encoded question based on that dictionary.\nYour task is:\n- First, use the code dictionary to decode the encoded question.\n- Second, answer the decoded question.\n- Third, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question."""

code_rule = """- {rule}\n"""

example_rule_nocode = """## Input\n[question]: Indicates the question.\n\n## Output\n[solution]: Indicates the process of answering the question.\n[answer]: Indicates the answer. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question.\n"""

example_rule_code = """## Input\n[question]: Indicates the question after encoding. Some of the words in the question are encoded using the code dictionary. \n\n## Output\n[solution]: Indicates the process of decoding and the process of answering the question.\n[answer]: Indicates the answer. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question.\n"""

example_num_explain = """I will give you {e_num} example(s), please give [solution] and [answer] based on the example(s):\n"""

examples = [
    {
        "text": "Write a function to multiply two integers",
        "options": ".\nYour code should pass these tests: \n['assert multiply_int(10,20)==200', 'assert multiply_int(5,10)==50', 'assert multiply_int(4,8)==32']\n\nThere may be libraries you need to import: \n[]\n",
        "en_words": ["Write", "function", "multiply"],
        "solution": "To multiply two integers, we can simply use the built-in multiplication operator in Python. However, since the task asks to write a function to do this, here's a simple function that takes two integers as input and returns their product:\n\n```python\ndef multiply_int(a, b):\n    return a * b\n```",
        "answer": "def multiply_int(a, b):\n    return a * b\n"
    },
    {
        "text": "Write a python function to find the length of the longest sublists",
        "options": ".\nYour code should pass these tests: \n['assert Find_Max_Length([[1],[1,4],[5,6,7,8]]) == 4', 'assert Find_Max_Length([[0,1],[2,2,],[3,2,1]]) == 3']\n\nThere may be libraries you need to import: \n[]\n",
        "en_words": ["Write", "python", "find", "length", "longest"],
        "solution": "To solve this problem, we need to define a function that takes a list of lists as an input and returns the length of the longest sublist. Here's a simple function that uses the built-in `max` function with a generator expression to find the length of the longest sublist.```python\ndef Find_Max_Length(lst):  \n    maxLength = max(len(x) for x in lst )\n    return maxLength\n```",
        "answer": "def Find_Max_Length(lst):  \n    maxLength = max(len(x) for x in lst )\n    return maxLength\n"
    },
    {
        "text": "Write a python function to interchange the first and last elements in a list",
        "options": ".\nYour code should pass these tests: \n['assert swap_List([12, 35, 9, 56, 24]) == [24, 35, 9, 56, 12]', 'assert swap_List([1, 2, 3]) == [3, 2, 1]']\n\nThere may be libraries you need to import: \n[]\n",
        "en_words": ["Write", "python", "to", "interchange", "first", "elements"],
        "solution": "To solve this problem, we can create a function that takes a list as an argument and swaps the first and last elements. Here's a simple function that accomplishes this:\n\n```python\ndef swap_List(input_list):\n    if len(input_list) > 1:\n        input_list[0], input_list[-1] = input_list[-1], input_list[0]\n    return input_list\n```",
        "answer": "def swap_List(input_list):\n    if len(input_list) > 1:\n        input_list[0], input_list[-1] = input_list[-1], input_list[0]\n    return input_list\n"
    },
    {
        "text": "Write a function to compare the value of two numbers",
        "options": ".\nYour code should pass these tests: \n['assert compare(2, 4) == 4', 'assert compare(-1, 12) == 12', 'assert compare(6, 5) == 6']\n\nThere may be libraries you need to import: \n[]\n",
        "en_words": ["a", "function", "compare", "value"],
        "solution": "Here's a simple Python function to compare two numbers:\n\n```python\ndef compare(a, b):\n    if a >= b:\n        return a\n    elif a < b:\n        return b\n```",
        "answer": "def compare(a, b):\n    if a >= b:\n        return a\n    elif a < b:\n        return b"
    },
    {
        "text": "Write a function to calculate the number of a in a string, regardless of size",
        "options": ".\nYour code should pass these tests: \n['assert count_a(\'dasdascewcascawfgd\') == 4', 'assert count_a(\'cascdsgvsvdvasvas\') == 3', 'assert count_a(\'hiuoawndannkasdnkl\') == 3']\nThere may be libraries you need to import: \n[]\n",
        "en_words": ["a", "to", "the", "of", "regardless"],
        "solution": "Here's a function for evaluating strings that contain the letter a (case-insensitive):\n\n```python\ndef count_a(input_string):\n    if not isinstance(input_string, str):\n        raise ValueError(\"Input must be a string.\")\n    # Convert string to lowercase and count 'a'\n    return input_string.lower().count('a')\n```",
        "answer": "def count_a(input_string):\n    if not isinstance(input_string, str):\n        raise ValueError(\"Input must be a string.\")\n    # Convert string to lowercase and count 'a'\n    return input_string.lower().count('a')"
    }
]

end = """
Now, I will give you a question, please answer it: 
[question]
{question}
Your code should pass these tests: \n{test_list}
There may be libraries you need to import: \n{test_imports}"""

simple_template = """Answer the following question. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question.{rule} Think step by step before answering.\n\n{examples}### Question:\n{question}\nYour code should pass these tests: \n{test_list}
There may be libraries you need to import: \n{test_imports}"""

decode_template = """I'll give you a code dictionary and a encoded question based on that dictionary. Your task is to decode the encoded question(**Don't answer the decoded question**). The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is the decoded question.{rule} Think step by step before answering.\n\n{examples}### Encoded Question:\n{question}"""


def encode_w(w, dic):
    """
    Encoding a word
    """
    res = []
    ori = []
    for i in w:
        i = i.upper()
        if i in list(dic.keys()):
            res.append(dic.get(i))
            ori.append(i)
        else:
            res.append(i)
    return '|'.join(res), res, ori


def encode_text_special(text, e_words, dic, options, solution, answer):
    """
    Constructing a shot after encoding a question
    """
    output_text = ""
    w_list = []
    words = text.strip().split(" ")
    ex_list = []
    a_list = []
    for word in words:
        if word in e_words:
            e_words.remove(word)
            w_list.append(encode_w(word, dic)[0])
            encoded_segment = encode_w(word, dic)[0]
            encoded_char = encode_w(word, dic)[1]
            ori_char = encode_w(word, dic)[2]
            ex_text = f"encoded segment is \"{encoded_segment}\"\n"
            for i in range(len(encoded_char)):
                ex_text += f"{encoded_char[i]} -> {ori_char[i]}\n"
            ex_text += f"This forms the word: \"{word.upper()}\""
            ex_list.append(ex_text)
            # ✨🌊 -> A\n✨🌊✨ -> R\n✨ -> E\nThis forms the word: "ARE"
            a_list.append(word.upper())
        else:
            w_list.append(word)
            a_list.append(word)
    output_text += "[question]\n"
    output_text += f"{' '.join(w_list)}{options}\n"
    output_text += "[solution]\n"
    output_text += "Let's first decode the encoded segments provided in the text:\nThe encoded text before decode is:\n"
    output_text += f"\"{' '.join(w_list)}{options}\"\n"


    pre_list = ["st", "nd", "rd", "th"]
    for i in range(len(ex_list)):
        if (i+1) % 10 <= 3:
            output_text += f"The {i+1}{pre_list[i]} {ex_list[i]}\n"
        else:
            output_text += f"The {i+1}{pre_list[3]} {ex_list[i]}\n"
    output_text += f"Therefore, the original question after decode is : \\\"{' '.join(a_list)}{options}\\\"\n"
    output_text += f"{solution}\n"
    output_text += f"[answer]\nAnswer: {answer}\n\n"
    return output_text


def text_special(text, options, solution, answer):
    """
    Constructing a single shot
    """
    output_text = ""
    output_text += "[question]\n"
    output_text += f"{text+options}\n"
    output_text += f"[solution]\n{solution}\n"
    output_text += f"[answer]\nAnswer: {answer}\n\n"
    return output_text


def common(example_num, code_if, dic):
    """
    The main code, which constructs the prompt for the main experiment
    """
    output_text = ""
    if code_if:
        output_text += f"{guide_code}\n{code_rule}\n\n"
    else:
        output_text += f"{guide_nocode}\n\n\n"
    if code_if and example_num > 0:
        output_text += f"{example_rule_code}\n"
    elif not code_if and example_num > 0:
        output_text += f"{example_rule_nocode}\n"
    if example_num > 0:
        output_text += f"{example_num_explain.format(e_num=example_num)}\n"
    for idx, item in enumerate(examples):
        if idx == example_num:
            break
        output_text += f"### Example {idx+1}:\n"
        if code_if:
            output_text += encode_text_special(item["text"], copy.deepcopy(item["en_words"]), dic, item["options"], item["solution"], item["answer"])
        else:
            output_text += text_special(item["text"], item["options"], item["solution"], item["answer"])
    output_text += f"{end}\n"
    return output_text.strip()


def simple_text_special(code_if, text, options, solution, answer, e_words, dic):
    """
    Pick a question to construct a simple shot
    """
    output_text = ""
    if not code_if:
        output_text += f"{text+options}\n\n"
        output_text += f"{solution}\n"
        output_text += f"Answer: {answer}\n\n"
    else:
        w_list = []
        words = text.strip().split(" ")
        ex_list = []
        a_list = []
        for word in words:
            if word in e_words:
                e_words.remove(word)
                w_list.append(encode_w(word, dic)[0])
                encoded_segment = encode_w(word, dic)[0]
                encoded_char = encode_w(word, dic)[1]
                ori_char = encode_w(word, dic)[2]
                ex_text = f"encoded segment is \"{encoded_segment}\"\n"
                for i in range(len(encoded_char)):
                    ex_text += f"{encoded_char[i]} -> {ori_char[i]}\n"
                ex_text += f"This forms the word: \"{word.upper()}\""
                ex_list.append(ex_text)
                a_list.append(word.upper())
            else:
                w_list.append(word)
                a_list.append(word)
        output_text += f"{' '.join(w_list)}{options}\n\n"
        output_text += "Let's first decode the encoded segments provided in the text:\nThe encoded text before decode is:\n"
        output_text += f"\"{' '.join(w_list)}{options}\"\n"

        pre_list = ["st", "nd", "rd", "th"]
        for i in range(len(ex_list)):
            if (i+1) % 10 <= 3:
                output_text += f"The {i+1}{pre_list[i]} {ex_list[i]}\n"
            else:
                output_text += f"The {i+1}{pre_list[3]} {ex_list[i]}\n"
        output_text += f"Therefore, the original question after decode is : \\\"{' '.join(a_list)}{options}\\\"\n"
        output_text += solution + "\n"
        output_text += f"Answer: {answer}\n\n"
    return output_text 


def simple_common(example_num, code_if, dic):
    """
    The main code block, used to construct the simpler prompt for the main experiment
    """
    example_text = ""
    for idx, item in enumerate(examples):
        if idx == example_num:
            break
        example_text += f"### Example {idx+1}:\n"
        example_text += simple_text_special(code_if, item["text"], item["options"], item["solution"], item["answer"], copy.deepcopy(item["en_words"]), dic)
    if example_num > 0:
        example_text = f"I will give you {example_num} example(s), please give me answer based on the example(s):\n" + example_text
        example_text += "\n"
    if code_if:
        output_text = simple_template.format(rule="\n\n{rule}", examples=example_text, question="{question}", test_imports="{test_imports}", test_list="{test_list}")
    else:
        output_text = simple_template.format(rule="", examples=example_text, question="{question}", test_imports="{test_imports}", test_list="{test_list}")
    return output_text.strip()


def decode_text_special(text, e_words, dic):
    """
    Pick a question to build a shot that is simply used to test decode
    """
    output_text = ""
    w_list = []
    words = text.strip().split(" ")
    ex_list = []
    a_list = []
    for word in words:
        if word in e_words:
            e_words.remove(word)
            w_list.append(encode_w(word, dic)[0])
            encoded_segment = encode_w(word, dic)[0]
            encoded_char = encode_w(word, dic)[1]
            ori_char = encode_w(word, dic)[2]
            ex_text = f"encoded segment is \"{encoded_segment}\"\n"
            for i in range(len(encoded_char)):
                ex_text += f"{encoded_char[i]} -> {ori_char[i]}\n"
            ex_text += f"This forms the word: \"{word.upper()}\""
            ex_list.append(ex_text)
            a_list.append(word.upper())
        else:
            w_list.append(word)
            a_list.append(word)
    output_text += f"{' '.join(w_list)}\n\n"
    output_text += "Let's decode the encoded segments provided in the text:\nThe encoded text before decode is:\n"
    output_text += f"\"{' '.join(w_list)}\"\n"

    pre_list = ["st", "nd", "rd", "th"]
    for i in range(len(ex_list)):
        if (i+1) % 10 <= 3:
            output_text += f"The {i+1}{pre_list[i]} {ex_list[i]}\n"
        else:
            output_text += f"The {i+1}{pre_list[3]} {ex_list[i]}\n"
    output_text += f"Therefore, the original question after decode is : \\\"{' '.join(a_list)}\\\"\n"
    output_text += f"Answer: {' '.join(a_list)}\n\n"
    return output_text 


def decode_common(example_num, dic):
    """
    The main block of code that constructs the prompt for the decode experiment.
    """
    example_text = ""
    for idx, item in enumerate(examples):
        if idx == example_num:
            break
        example_text += f"### Example {idx+1}:\n"
        example_text += decode_text_special(item["text"], copy.deepcopy(item["en_words"]), dic)
    if example_num > 0:
        example_text = f"I will give you {example_num} example(s), please give me answer based on the example(s):\n" + example_text
        example_text += "\n"
    output_text = decode_template.format(rule="\n\n{rule}", examples=example_text, question="{question}")
    return output_text.strip()


if __name__ == "__main__":
    emoj_shuffle_dict = {"A": "😀🍎🚗", "B": "🐶🌟📚", "C": "🌈🍀🚀", "D": "🐱🍉🏀", "E": "🍔🎉🎈", "F": "🌸🍩🏰",
                "G": "🦋🍇⚽", "H": "🍕🎂🏝️", "I": "🍦🎁🎧", "J": "🐸🍒🏆", "K": "🦄🍓🎮", "L": "🐰🍍📷",
                "M": "🌹🍌🎨", "N": "🐼🍎🎤", "O": "🍉🎵📚", "P": "🌼🍇🎬", "Q": "🐢🍓🎯", "R": "🍒🎸📱",
                "S": "🌻🍍🎲", "T": "🐯🍌🎮", "U": "🍓🎹📖", "V": "🌺🍉🎥", "W": "🐳🍎🎭", "X": "🍍🎤📡",
                "Y": "🐥🍇🎨", "Z": "🌵🍒🎮", "0": "🦁🍄🎈", "1": "🐘🌸🎨", "2": "🦊🍉🌟",
                "3": "🦋🍇🎵", "4": "🦄🍓🎲", "5": "🐼🍍🎯", "6": "🦉🍌🎤", "7": "🦇🍒🎧",
                "8": "🦄🍂🎥", "9": "🦍🍁🎮"}
    print(simple_common(example_num=5, code_if=True, dic=emoj_shuffle_dict))
    # print(decode_common(example_num=3, dic=emoj_shuffle_dict))
