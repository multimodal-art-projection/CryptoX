# data_construct/prompt_create/prompt_mmlu.py
# prompt中的部件
answer_range = {"digit": "where YOUR_ANSWER is one of 1234", "alpha": "where YOUR_ANSWER is one of ABCD", "final": "where YOUR_ANSWER consists of 2 letter or number"}

guide_nocode = """You are a good problem-solver, I'll give you a question and a answer conversion rule.\nYour task is:\n- First, answer the question using a answer conversion rule.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is one of {answer_range}.\n"""

guide_code = """You are a good problem-solver, I'll give you a code dictionary, a answer conversion rule and a encoded question based on that dictionary.\nYour task is:\n- First, use the code dictionary to decode the encoded question.\n- Second, answer the decoded question.\n- Third, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is one of {answer_range}.\n"""

guide_forall = """You are a good problem-solver, I'll give you a question.\nYour task is:\n"""

guide_nocode = """You are a good problem-solver, I'll give you a question.\nYour task is:\n"""

guide_code = """You're a decoding expert and smart problem-solver, I'll give you a code dictionary and a encoded question based on that dictionary.\n"""

guide_step = {"code_rule": "use the code dictionary to decode the encoded question.", "answer_question": "answer the question.", "num_get": "use 1-26 for A-Z to get the number corresponding to the correct option.", "cat_answer": "stitch together the number obtained in the previous step and the first alphanumeric character in the specific answer corresponding to the correct option (**except for the option**) to get the final answer.", "output_format": "output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), {answer_range}."}

code_rule = """- {rule}\n"""

hop_rule_2 = """- Answer conversion rule is: 
the answer is A you should answer B
the answer is B you should answer C
the answer is C you should answer D
the answer is D you should answer A\n"""

hop_rule_3 = """- Answer conversion rule is: 
the answer is A you should change it to B, and finally answer 2
the answer is B you should change it to C, and finally answer 3
the answer is C you should change it to D, and finally answer 4
the answer is D you should change it to A, and finally answer 1\n"""

example_rule_nocode = """## Input\n[question]: Indicates the question.\n\n## Output\n[solution]: Indicates the process of answering the question.\n[answer]: Indicates the answer. The last line of your response should be of the following format: 'Answer: $YOUR_ANSWER' (without quotes), {answer_range}. \n"""

example_rule_code = """## Input\n[question]: Indicates the question after encoding. Some of the words in the question are encoded using the code dictionary. \n\n## Output\n[solution]: Indicates the process of decoding and the process of answering the question.\n[answer]: Indicates the answer. The last line of your response should be of the following format: 'Answer: $YOUR_ANSWER' (without quotes), {answer_range}. \n"""

example_num_explain = """I will give you {e_num} example(s), please give [solution] and [answer] based on the example(s):\n"""

examples = [
    {
        "text": "Which of the following is the body cavity that contains the pituitary gland",
        "options": "?\nA) Abdominal\nB) Cranial\nC) Pleural\nD) Spinal\n",
        "en_words": ["Which", "the", "following", "body", "contains"],
        "solution1": "The pituitary gland is located in the cranial cavity, specifically in a small depression of the sphenoid bone called the sella turcica. The correct answer is: B",
        "answer1": "B",
        "solution2": "The pituitary gland is located in the cranial cavity, specifically in a small depression of the sphenoid bone called the sella turcica. The correct answer is: B. Option B corresponds to the number 2 on a scale 1-26, and the final answer is 2.",
        "answer2": "2",
        "solution3": "The pituitary gland is located in the cranial cavity, specifically in a small depression of the sphenoid bone called the sella turcica. The correct answer is: B. Option B corresponds to the number 2 on a scale 1-26. Because the first alphanumeric character of \"Cranial\" is \"C\", the number obtained in the previous step is concatenated with it to get the final answer 2C.",
        "answer3": "2C"
    },
    {
        "text": "Which of these songs was a Top 10 hit for the rock band The Police",
        "options": "?\nA) 'Radio Ga-Ga'\nB) 'Ob-la-di Ob-la-da'\nC) 'De Do Do Do De Da Da Da'\nD) 'In-a-Gadda-Da-Vida'\n",
        "en_words": ["of", "songs", "Top", "hit", "rock"],
        "solution1": "The correct answer is C) 'De Do Do Do De Da Da Da'. This song by The Police was a hit in 1980, reaching the Top 10 on the Billboard Hot 100 in the U.S.",
        "answer1": "C",
        "solution2": "The correct answer is C) 'De Do Do Do De Da Da Da'. This song by The Police was a hit in 1980, reaching the Top 10 on the Billboard Hot 100 in the U.S. Option C corresponds to the number 3 on a scale 1-26, and the final answer is 3.",
        "answer2": "3",
        "solution3": "The correct answer is C) 'De Do Do Do De Da Da Da'. This song by The Police was a hit in 1980, reaching the Top 10 on the Billboard Hot 100 in the U.S. Option C corresponds to the number 3 on a scale 1-26. Because the first alphanumeric character of 'De Do Do Do De Da Da Da' is \"D\", the number obtained in the previous step is concatenated with it to get the final answer 3D.",
        "answer3": "3D"
    },
    {
        "text": "The population of the city where Michelle was born is 145,826. What is the value of the 5 in the number 145,826",
        "options": "?\nA) 5 thousands\nB) 5 hundreds\nC) 5 tens\nD) 5 ones\n",
        "en_words": ["population", "the", "where", "was", "value"],
        "solution1": "The digit 5 is in the thousands place, representing 5,000. So, the value of the 5 in the number 145,826 is 5 thousands. The correct answer is: A",
        "answer1": "A",
        "solution2": "The digit 5 is in the thousands place, representing 5,000. So, the value of the 5 in the number 145,826 is 5 thousands. The correct answer is: A. Option A corresponds to the number 1 on a scale 1-26, and the final answer is 1.",
        "answer2": "1",
        "solution3": "The digit 5 is in the thousands place, representing 5,000. So, the value of the 5 in the number 145,826 is 5 thousands. The correct answer is: A. Option A corresponds to the number 1 on a scale 1-26. Because the first alphanumeric character of \"5 thousands\" is \"5\", the number obtained in the previous step is concatenated with it to get the final answer 15.",
        "answer3": "15"
    },
    {
        "text": "Which of the following is considered an acid anhydride",
        "options": "?\nA) HCl\nB) H2SO3\nC) SO2\nD) Al(NO3)3\n",
        "en_words": ["following", "considered", "acid", "anhydride"],
        "solution1": "The correct answer is C) SO2. \nAn acid anhydride is a compound that forms an acid when it reacts with water. In this case, SO2 (sulfur dioxide) is an acid anhydride because it reacts with water to form sulfurous acid (H2SO3).",
        "answer1": "C",
        "solution2": "The correct answer is C) SO2. \nAn acid anhydride is a compound that forms an acid when it reacts with water. In this case, SO2 (sulfur dioxide) is an acid anhydride because it reacts with water to form sulfurous acid (H2SO3). Option C corresponds to the number 3 on a scale 1-26, and the final answer is 3.",
        "answer2": "3",
        "solution3": "The correct answer is C) SO2. \nAn acid anhydride is a compound that forms an acid when it reacts with water. In this case, SO2 (sulfur dioxide) is an acid anhydride because it reacts with water to form sulfurous acid (H2SO3). Option C corresponds to the number 3 on a scale 1-26. Because the first alphanumeric character of \"SO2\" is \"S\", the number obtained in the previous step is concatenated with it to get the final answer 3S.",
        "answer3": "3S"
    },
    {
        "text": "An auditor traces the serial numbers on equipment to a nonissuer's subledger. Which of the following management assertions is supported by this test",
        "options": "?\nA) Valuation and allocation\nB) Completeness\nC) Rights and obligations\nD) Presentation and disclosure\n",
        "en_words": ["auditor", "serial", "equipment", "Which", "following", "assertions", "supported"],
        "solution1": "The correct answer is B) Completeness. \nTracing serial numbers to the subledger ensures that all equipment is recorded, which supports the completeness assertion.",
        "answer1": "B",
        "solution2": "The correct answer is B) Completeness. \nTracing serial numbers to the subledger ensures that all equipment is recorded, which supports the completeness assertion. Option B corresponds to the number 2 on a scale 1-26, and the final answer is 2.",
        "answer2": "2",
        "solution3": "The correct answer is B) Completeness. \nTracing serial numbers to the subledger ensures that all equipment is recorded, which supports the completeness assertion. Option B corresponds to the number 2 on a scale 1-26. Because the first alphanumeric character of \"Completeness\" is \"C\", the number obtained in the previous step is concatenated with it to get the final answer 2C.",
        "answer3": "2C"
    }
]

end = """
Now, I will give you a question, please answer it: 
[question]
{question}
{options}"""

simple_template = """Answer the following question. {step}The last line of your response should be of the following format: 'Answer: $YOUR_ANSWER' (without quotes), {answer_range}.{rule} Think step by step before answering.\n\n{examples}### Question:\n{question}\n\n{options}"""

decode_template = """I'll give you a code dictionary and a encoded question based on that dictionary. Your task is to decode the encoded question(**Don't answer the decoded question**). The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is the decoded question.{rule} Think step by step before answering.\n\n{examples}### Encoded Question:\n{question}"""

cr_decode_template = """I'll give you a code dictionary and a encoded question based on that dictionary. Your task is to decode the encoded question(**Don't answer the decoded question**). The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is the decoded question.{rule} Think step by step before answering. ### Encoded Question:\n{question}"""

cr_simple_template = """Please answer the decoded question above with the options below. {step}The last line of your response should be of the following format: 'Answer: $YOUR_ANSWER' (without quotes), {answer_range}. Think step by step before answering. ### Options:\n{options}"""


def encode_w(w, dic):
    """
    对单个单词进行编码
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
    对单个问题进行编码，然后生成一个shot
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
    构造单个shot
    """
    output_text = ""
    output_text += "[question]\n"
    output_text += f"{text+options}\n"
    output_text += f"[solution]\n{solution}\n"
    output_text += f"[answer]\nAnswer: {answer}\n\n"
    return output_text


def change_answer(answer, hopnum):
    """
    根据不同的hop修改答案
    """
    if hopnum == 1:
        return answer
    elif hopnum == 2:
        answer_num = (ord(answer) - ord('A') + 1) % 4 + ord('A')
        return chr(answer_num)
    else:
        answer_num = (ord(answer) - ord('A') + 1) % 4 + ord('1')
        return int(chr(answer_num))
    
    
def change_solution(sollution, answer, hopnum):
    """
    根据不同的hop修改solution
    """
    if hopnum == 1:
        return sollution
    elif hopnum == 2:
        answer_text = " According to the answer conversion rule, if the answer is {true_answer}, I should answer {new_answer}. The correct answer is {new_answer}"
        answer_num = (ord(answer) - ord('A') + 1) % 4 + ord('A')
        return sollution + answer_text.format(true_answer=answer, new_answer=chr(answer_num))
    else:
        answer_text = " According to the answer conversion rule, if the answer is {true_answer}, I should change it to {new_answer_1}, and finally answer {new_answer_2}. The correct answer is {new_answer_2}"
        answer_num_1 = (ord(answer) - ord('A') + 1) % 4 + ord('A')
        answer_num_2 = (ord(answer) - ord('A') + 1) % 4 + ord('1')
        return sollution + answer_text.format(true_answer=answer, new_answer_1=chr(answer_num_1), new_answer_2=chr(answer_num_2))
    

def collect_rule(g_list):
    """
    构造task_rule
    """
    length = len(g_list)
    num_dic = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth", 6: "Sixth"}
    text = ""
    for i in range(length):
        text+=f"- {num_dic[i+1]}, {g_list[i]}\n"
    return text


def common(example_num, code_if, dic, hop_num = 1):
    """
    主代码，构造主实验用的prompt
    """
    output_text = ""
    guide_list = []
    if code_if:
        output_text += f"{guide_code}"
    else:
        output_text += f"{guide_nocode}"
    if code_if:
        guide_list.append(guide_step["code_rule"])
    guide_list.append(guide_step["answer_question"])
    if hop_num == 1:
        ar = answer_range["alpha"]
    elif hop_num == 2:
        guide_list.append(guide_step["num_get"])
        ar = answer_range["digit"]
    elif hop_num == 3:
        guide_list.append(guide_step["num_get"])
        guide_list.append(guide_step["cat_answer"])
        ar = answer_range["final"]
    guide_list.append(guide_step["output_format"].format(answer_range=ar))
    output_text += f"{collect_rule(guide_list)}"
    if code_if:
        output_text += f"{code_rule}\n\n"
        if example_num > 0:
            output_text += f"{example_rule_code.format(answer_range=ar)}\n"
    elif not code_if:
        if example_num > 0:
            output_text += f"\n\n{example_rule_nocode.format(answer_range=ar)}\n"
        else:
            output_text += f"\n\n"
    if example_num > 0:
        output_text += f"{example_num_explain.format(e_num=example_num)}\n"
    for idx, item in enumerate(examples):
        if idx == example_num:
            break
        output_text += f"### Example {idx+1}:\n"
        if code_if:
            output_text += encode_text_special(item["text"], item["en_words"], dic, item["options"], item[f"solution{hop_num}"], item[f"answer{hop_num}"])
        else:
            output_text += text_special(item["text"], item["options"], item[f"solution{hop_num}"], item[f"answer{hop_num}"])
    output_text += f"{end}\n"
    return output_text.strip()


def simple_text_special(code_if, text, options, solution, answer, e_words, dic):
    """
    选取一个问题构建一个简单的shot
    """
    output_text = ""
    if not code_if:
        output_text += f"{text+options}\n"
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
        output_text += f"{' '.join(w_list)}{options}\n"
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


def simple_common(example_num, code_if, dic, hop_num = 1):
    """
    主代码块，用于构造主实验较简单的prompt
    """
    example_text = ""
    for idx, item in enumerate(examples):
        if idx == example_num:
            break
        example_text += f"### Example {idx+1}:\n"
        example_text += simple_text_special(code_if, item["text"], item["options"], item[f"solution{hop_num}"], item[f"answer{hop_num}"], item["en_words"], dic)
    if hop_num == 1:
        ar = answer_range["alpha"]
        step = ""
    elif hop_num == 2:
        ar = answer_range["digit"]
        step = f"{guide_step['num_get'][0].upper() + guide_step['num_get'][1:]} "
    elif hop_num == 3:
        ar = answer_range["final"]
        step = f"{guide_step['num_get'][0].upper() + guide_step['num_get'][1:]} {guide_step['cat_answer'][0].upper() + guide_step['cat_answer'][1:]} "
    if example_num > 0:
        example_text = f"I will give you {example_num} example(s), please give me answer based on the example(s):\n" + example_text
        example_text += "\n"
    if code_if:
        output_text = simple_template.format(step=step, answer_range=ar, rule="\n\n{rule}", examples=example_text, question="{question}", options="{options}")
    else:
        output_text = simple_template.format(step=step, answer_range=ar, rule="", examples=example_text, question="{question}", options="{options}")
    return output_text.strip()


def decode_text_special(text, e_words, dic):
    """
    选取一个问题构建一个简单用于测试decode的shot
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
    主代码块，用于构造测decode实验的prompt
    """
    example_text = ""
    for idx, item in enumerate(examples):
        if idx == example_num:
            break
        example_text += f"### Example {idx+1}:\n"
        example_text += decode_text_special(item["text"], item["en_words"], dic)
    if example_num > 0:
        example_text = f"I will give you {example_num} example(s), please give me answer based on the example(s):\n" + example_text
        example_text += "\n"
    output_text = decode_template.format(rule="\n\n{rule}", examples=example_text, question="{question}")
    return output_text.strip()


def cr_simple(dic, hop_num=1):
    """
    主代码块，用于构造cr实验的prompt
    """
    output_text = list()
    if hop_num == 1:
        ar = answer_range["alpha"]
        step = ""
    elif hop_num == 2:
        ar = answer_range["digit"]
        step = f"{guide_step['num_get'][0].upper() + guide_step['num_get'][1:]} "
    elif hop_num == 3:
        ar = answer_range["final"]
        step = f"{guide_step['num_get'][0].upper() + guide_step['num_get'][1:]} {guide_step['cat_answer'][0].upper() + guide_step['cat_answer'][1:]} "
    output_text.append(cr_decode_template.format(rule="\n\n{rule}", question="{question}").strip())
    output_text.append(cr_simple_template.format(step=step, answer_range=ar, options="{options}").strip())
    return output_text


if __name__ == "__main__":
    emoj_shuffle_dict = {"A": "😀🍎🚗", "B": "🐶🌟📚", "C": "🌈🍀🚀", "D": "🐱🍉🏀", "E": "🍔🎉🎈", "F": "🌸🍩🏰",
                "G": "🦋🍇⚽", "H": "🍕🎂🏝️", "I": "🍦🎁🎧", "J": "🐸🍒🏆", "K": "🦄🍓🎮", "L": "🐰🍍📷",
                "M": "🌹🍌🎨", "N": "🐼🍎🎤", "O": "🍉🎵📚", "P": "🌼🍇🎬", "Q": "🐢🍓🎯", "R": "🍒🎸📱",
                "S": "🌻🍍🎲", "T": "🐯🍌🎮", "U": "🍓🎹📖", "V": "🌺🍉🎥", "W": "🐳🍎🎭", "X": "🍍🎤📡",
                "Y": "🐥🍇🎨", "Z": "🌵🍒🎮", "0": "🦁🍄🎈", "1": "🐘🌸🎨", "2": "🦊🍉🌟",
                "3": "🦋🍇🎵", "4": "🦄🍓🎲", "5": "🐼🍍🎯", "6": "🦉🍌🎤", "7": "🦇🍒🎧",
                "8": "🦄🍂🎥", "9": "🦍🍁🎮"}
    # print(simple_common(example_num=3, code_if=True, dic=emoj_shuffle_dict, hop_num = 3))
    print(cr_simple(dic=emoj_shuffle_dict, hop_num=3))