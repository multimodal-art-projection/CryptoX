# eval/metric_lib/metric_mmlu.py
# python自带的库
import re
import asyncio


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


def judge_correctness_mmlu(problem, real_answer, generated_answer):
    """
    判断mmlu数据集的答案的正确性
    """
    generated_answer = normalize_response(generated_answer)
    pos = generated_answer.lower().rfind("answer")
    if pos == -1:
        return False
    generated_answer = generated_answer[pos:]
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*(.*)"
    match_for_generated_answer = re.findall(ANSWER_PATTERN_MULTICHOICE, generated_answer)
    extracted_answer = match_for_generated_answer[-1] if match_for_generated_answer else None
    if extracted_answer is None:
        return 0
    score = int(str(extracted_answer).replace("$", " ").replace(" ", "").replace(".", "").replace("(", "").replace(")", "") == str(real_answer).replace("$", " ").replace(" ", "").replace(".", "").replace("(", "").replace(")", ""))
    return score


async def judge_correctness(problem, real_answer, generated_answer):
    """
    异步版本的用于判断正确性的请求函数，使用运行中的事件循环
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, judge_correctness_mmlu, problem, real_answer, generated_answer)
    return result


if __name__ == '__main__':
    problem="A right cylindrical tank with circular bases is being filled with water at a rate of $20\\pi$ cubic meters per hour. As the tank is filled, the water level rises four meters per hour. What is the radius of the tank, in meters? Express your answer in simplest radical form."
    real_answer="B"
    generated_answer="To begin, let's decode the encoded question using the provided code dictionary.\n\n- 🍦🎁🎧 translates to 'I'\n- 🌻🍍🎲 translates to 'S'\n- 🐯🍌🎮 translates to 'T'\n- 🍕🎂🏝️ translates to 'H'\n- 🍔🎉🎈 translates to 'E'\n- 😀🍎🚗 translates to 'A'\n- 🌹🍌🎨 translates to 'M'\n- 🌼🍇🎬 translates to 'P'\n- 🍉🎵📚 translates to 'O'\n- 🍓🎹📖 translates to 'U'\n- 🌈🍀🚀 translates to 'C'\n- 🐰🍍📷 translates to 'L'\n- 🍒🎸📱 translates to 'R'\n- 🦋🍇⚽ translates to 'G'\n\nNow decode the encoded question:\n\n\"You translate IS THE TEAM direct ATTEMPT TO formally or informally manage ETHICAL ISSUES or problems, THAT RELATE TO specific policies, practices and programmes.\"\n\nDecoded Question: Is the team direct attempt to formally or informally manage ethical issues or problems, that relate to specific policies, practices and programmes.\n\nNow, examine the multiple-choice answers to determine what best matches this description:\n\nA) Corporate social responsibility - Related but not primarily about management of ethical issues or problems.\nB) Business ethics management - Focuses on managing ethical issues or problems.\nC) Sustainability - Related to ethical concerns but not specifically about management.\nD) Environmental management - Related to specific issues but more focused on environmental concerns.\n\nThe decoded question pertains to the management of ethical issues, making the most suitable answer:\n\nAnswer: B"
    test_answer = "Let's start by decoding the encoded message using the provided code dictionary:\n\n1. 🍉🎵📚 -> O\n2. 🐼🍎🎤 -> N\n3. 🍔🎉🎈 -> E\n4. 🍕🎂🏝️ -> H\n5. 🍓🎹📖 -> U\n6. 🐱🍉🏀 -> D\n7. 🍒🎸📱 -> R\n8. 🌺🍉🎥 -> V\n9. 🌻🍍🎲 -> S\n10. 🐯🍌🎮 -> T\n11. 🐥🍇🎨 -> Y\n12. 🌹🍌🎨 -> M\n13. 🐳🍎🎭 -> W\n\nNow, decode the question step-by-step:\n\n- 🍉🎵📚|🐼🍎🎤|🍔🎉🎈 = ONE\n- 🍕🎂🏝️|🍓🎹📖|🐼🍎🎤|🐱🍉🏀|🍒🎸📱|🍔🎉🎈|🐱🍉🏀 = HUNDRED\n- 🐥🍇🎨|🍉🎵📚|🍓🎹📖|🍒🎸📱 = YOUR\n- 🍦🎁🎧|🐼🍎🎤|🌺🍉🎥|🍔🎉🎈|🌻🍍🎲|🐯🍌🎮|🍔🎉🎈|🐱🍉🏀 = INVESTED\n- 🍦🎁🎧|🐼🍎🎤|🌺🍉🎥|🍔🎉🎈|🌻🍍🎲|🐯🍌🎮|🌹🍌🎨|🍔🎉🎈|🐼🍎🎤|🐯🍌🎮 = INVESTMENT\n- 🐳🍎🎭|🍉🎵📚|🍒🎸📱|🐯🍌🎮|🍕🎂🏝️ = WORTH\n\nReconstructing the question:\n\n\"ONE HUNDRED years ago, YOUR great-great-grandmother INVESTED $100 at 5% yearly interest. What is the INVESTMENT WORTH today?\"\n\nNow, solve the question:\n\nThis is a classic compound interest problem. The formula for compound interest is:\n\n\\[ A = P \\times \\left(1 + \\frac{r}{100}\\right)^n \\]\n\nWhere:\n- \\( A \\) is the amount of money accumulated after n years, including interest.\n- \\( P \\) is the principal amount (initial investment).\n- \\( r \\) is the annual interest rate (in percentage terms).\n- \\( n \\) is the number of years the money is invested for.\n\nGiven:\n- \\( P = 100 \\),\n- \\( r = 5 \\),\n- \\( n = 100 \\).\n\n\\[ A = 100 \\times \\left(1 + \\frac{5}{100}\\right)^{100} \\]\n\\[ A = 100 \\times (1.05)^{100} \\]\n\nCalculating \\( (1.05)^{100} \\) approximately equals 13,150. \n\nSo, \\( A = 100 \\times 13,150 \\) which is approximately 13,000.\n\nThe answer to the question is: A) $13,000\n\nAnswer: A"
    test = "\nAnswer: B"
    print(judge_correctness_mmlu(real_answer,test))