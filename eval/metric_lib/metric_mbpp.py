# eval/metric_lib/metric_mbpp.py
# python自带的库
import multiprocessing


async def multichoice_openai(response, test):
    """
    异步请求对代码块能否正确执行test进行判断
    """
    try:
        response_specific = extract_code(response)
        score = run_tests(response_specific, test)
        return score
    except Exception as e:
        print(f"Error processing response: {response}\nError: {e}")
        return 0, "error"
    

def extract_code(text):
    """
    提取response中的代码块
    """
    start = text.rfind("def ")
    if start == -1:
        return ""
    end = text[start: ].find("return")
    end2 = text[start+end: ].find("\n")
    if end2 == -1:
        end2 = len(text[start+end: ])
    re = text[start: start + end + end2]
    return re


def run_tests(code, test_cases, timeout=3):
    """
    测试代码能否正确执行test_list
    """
    # 使用共享变量来跨进程共享 success_count 和 failure_count
    success_count = multiprocessing.Value('i', 0)  # 'i' 表示整数类型
    failure_count = multiprocessing.Value('i', 0)  # 'i' 表示整数类型

    def execute_code(test_case, success_count, failure_count):
        # 在此函数中执行代码
        # 执行函数定义代码
        exec(code, globals(), locals())
        
        if test_case.strip():  # 跳过空行
            try:
                exec(test_case, globals(), locals())
                with success_count.get_lock():  # 使用锁来确保修改安全
                    success_count.value += 1
            except AssertionError:
                with failure_count.get_lock():  # 使用锁来确保修改安全
                    failure_count.value += 1
                print(f"Test failed: {test_case}")
            except Exception as e:
                with failure_count.get_lock():  # 使用锁来确保修改安全
                    failure_count.value += 1
                print(f"Test failed due to error: {test_case}\nError: {e}")

    # 启动子进程
    processes = []
    for test_case in test_cases:
        process = multiprocessing.Process(target=execute_code, args=(test_case, success_count, failure_count))
        processes.append(process)
        process.start()

    # 等待所有子进程完成，限制超时
    for process in processes:
        process.join(timeout)
        if process.is_alive():
            print(f"Code execution timed out after {timeout} seconds")
            with failure_count.get_lock():  # 使用锁来确保修改安全
                failure_count.value += 1
            process.terminate()  # 强制终止子进程
            process.join()  # 确保进程被完全终止

    # 打印最终的测试结果
    return success_count.value / len(test_cases)


if __name__ == '__main__':
    # 模型生成的代码（假设这是从大模型获取的 response）
    response = """Let me solve this step by step.\n\n1) First, let's understand what a centered hexagonal number is:\n   * Centered hexagonal numbers are numbers that represent hexagons with a dot in the center and dots surrounding it in layers\n   * Each layer adds points in a hexagonal pattern\n   * The sequence goes: 1, 7, 19, 37, 61, 91, 127, 169, 217, 271, ...\n\n2) Looking at the pattern:\n   * From 1 to 7: difference is 6\n   * From 7 to 19: difference is 12\n   * From 19 to 37: difference is 18\n   * From 37 to 61: difference is 24\n   We can see each difference increases by 6\n\n3) Let's derive the formula:\n   * For the nth centered hexagonal number, each layer adds 6 more points than the previous layer\n   * The formula turns out to be: 3n(n-1) + 1\n   * Where n is the position in the sequence\n\n4) Let's write the function:\n\n```python\ndef centered_hexagonal_number(n):\n    return 3*n*(n-1) + 1\n```\n\n5) Verify the test cases:\n   * For n=10: 3*10*9 + 1 = 270 + 1 = 271 ✓\n   * For n=2: 3*2*1 + 1 = 6 + 1 = 7 ✓\n   * For n=9: 3*9*8 + 1 = 216 + 1 = 217 ✓\n\nAnswer: def centered_hexagonal_number(n): return 3*n*(n-1) + 1"""
    # 测试用例列表
    test_list = ["assert centered_hexagonal_number(10) == 271",
        "assert centered_hexagonal_number(2) == 7",
        "assert centered_hexagonal_number(9) == 217"
    ]
    # 运行测试
    response = extract_code(response)
    print(response)
    print(run_tests(response, test_list))
