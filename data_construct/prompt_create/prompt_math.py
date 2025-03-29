# data_construct/prompt_create/prompt_math.py
# Python Standard Library
import copy
# Components in prompt
guide_nocode = """You are a good math problem-solver, I'll give you a question.\nYour task is:\n- First, answer the question.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question."""

guide_code = """You are a good math problem-solver, I'll give you a code dictionary and a encoded question based on that dictionary.\nYour task is:\n- First, use the code dictionary to decode the encoded question.\n- Second, answer the decoded question.\n- Third, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question."""

code_rule = """- {rule}\n"""

example_rule_nocode = """## Input\n[question]: Indicates the question.\n\n## Output\n[solution]: Indicates the process of answering the question.\n[answer]: Indicates the answer. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question.\n"""

example_rule_code = """## Input\n[question]: Indicates the question after encoding.\n\n## Output\n[solution]: Indicates the process of decoding and the process of answering the question.\n[answer]: Indicates the answer. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question.\n"""

example_num_explain ="""I will give you {e_num} example(s), please give [solution] and [answer] based on the example(s):\n"""

examples = [
    {
        "text": "x=2, y=3, please calculate the value of x + 2y.",
        "options": "",
        "en_words": ["x=2,", "y=3,", "calculate", "value"],
        "solution": "x + 2y = 2 + 2 * 3 = 8, So the answer is 8.",
        "answer": "8"
    },
    {
        "text": "If a basket has 10 apples and you take 3 out, how many are left?",
        "options": "",
        "en_words": ["If", "basket", "apples", "you", "out,", "are"],
        "solution": "If a basket has 10 apples and you take 3 out, the total number of apples left in the basket is: 10 - 7 = 3.",
        "answer": "3"
    },
    {
        "text": "You have 12 cookies and divide them equally among 4 friends. How many cookies does each friend get?",
        "options": "",
        "en_words": ["You", "cookies", "divide", "among", "many", "does", "friend"],
        "solution": "If you have 12 cookies and divide them equally among 4 friends: 12 / 4 = 3",
        "answer": "3"
    },
    {
        "text": "There were 15 passengers on a bus, 5 people got off at the next stop, and 8 more people came up. How many people are in the car now?",
        "options": "",
        "en_words": ["There", "passengers", "people", "off", "came", "in"],
        "solution": "There were 15 passengers on the bus.\n5 people got off, so there are 10 passengers left.\nThen, 8 more people came on, making the total 18 passengers now.",
        "answer": "18"
    },
    {
        "text": "A little girl bought 4 pencils for 2 yuan each. How much did she spend in total?",
        "options": "",
        "en_words": ["little", "bought", "pencils", "yuan", "How", "did", "spend"],
        "solution": "Each pencil costs 2 yuan. She bought 4 pencils.\nSo, 4 * 2 = 8\nAnswer: She spent 8 yuan in total.",
        "answer": "8"
    }
]

end = """
Now, I will give you a question, please answer it: 
[question]
{question}"""

simple_template = """Answer the following question. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question.{rule} Think step by step before answering.\n\n{examples}### Question:\n{question}"""

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
    Encoding a question and constructing a shot
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
        output_text = simple_template.format(rule="\n\n{rule}", examples=example_text, question="{question}")
    else:
        output_text = simple_template.format(rule="", examples=example_text, question="{question}")
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
                "8": "🦄🍂🎥", "9": "🦍🍁🎮","$": "$", "(": "(", ")": ")", ",": ",", ".": ".", "\\": "\\", ">": "🎧📷🏰", "<": "📖🍍🏀", "=": "🍒🎈🍎", "_": "_", 
                "{": "{", "}": "}", "^": "🎲🍁🍇", "+": "🍄📚🚗", "-": "🐸🎁🌈", "?": "?", ";": ";", "\"": "\"", "'": "'", "/": "🐼🐘🦇", 
                "%": "🌟🦋🦄", "!": "🎹🌵🐯", "|": "🎭🎨🎵", "&": "🎤🍔🌟", "~": "🎮🐱🐸", "#": "#", "@": "@", "`": "`", "[": "[", "]": "]"}
    print(simple_common(example_num=5, code_if=False, dic=emoj_shuffle_dict))
    # print(decode_common(example_num=3, dic=emoj_shuffle_dict))