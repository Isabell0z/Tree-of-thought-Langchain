from langchain_experimental.tot.thought_generation import BaseThoughtGenerationStrategy

from langchain.prompts import PromptTemplate

import re
from langchain_core.prompts.base import BasePromptTemplate
from pydantic import Field
from typing import Any, Dict, List, Tuple

from uri_template import partial


def parse_game24_steps(output: str) -> list[str]:
    """
    从自然语言格式中提取有效的推理步骤。
    例如：1. 3 + 3 = 6 (left: 8 8 6)
    """
    lines = output.strip().splitlines()
    steps = []
    for line in lines:
        match = re.match(r"^\s*\d+\.\s*(.+?)\s*\(left:\s*(.*?)\s*\)", line)
        if match:
            expr = match.group(1).strip()
            rest = match.group(2).strip()
            steps.append(f"{expr} (left: {rest})")
    return steps

def get_game24_prompt(level: int) -> PromptTemplate:

    if level == 0:
        template = (
            "You are solving the Game 24 puzzle step by step.\n"
            "Given the numbers: {problem_description}, propose {n} possible next steps.\n"
            "Each step should combine **exactly two numbers** using one of the operations +, -, *, /,\n"
            "and show the result as a new number, along with the remaining numbers.\n"
            "Avoid invalid operations (e.g., division by zero).\n\n"
            "Format each step like this: \n"
            "a + b = c (left: remaining_number_1 remaining_number_2 c)\n\n"
            "→ where a and b are two numbers from the current list, c is the result,\n"
            "and 'left' contains the remaining numbers plus c.\n"
            "Do NOT use any variable names. Only use concrete numeric values.\n\n"
            "Possible next steps:\n"
        )
    elif level == 1:
        template = (
            "You are solving the Game 24 puzzle step by step.\n"
            "Given the numbers: {problem_description}, propose {n} possible next steps.\n"
            "Each step should combine **exactly two numbers** using one of the operations +, -, *, /,\n"
            "and show the result as a new number, along with the remaining numbers.\n"
            "Always use actual numbers. Do NOT use placeholders like 'remaining_number'.\n\n"
            "Avoid invalid operations (e.g., division by zero).\n\n"
            "Format each step like this: \n"
            "a + b = c (left: remaining_number,c)\n\n"
            "→ where a and b are two numbers from the current list, c is the result,\n"
            "and 'left' contains the remaining numbers plus c.\n"
            "Do NOT use any variable names. Only use concrete numeric values.\n\n"
            "Possible next steps:\n"
        )
    else: template = (
            "You are solving the Game 24 puzzle step by step.\n"
            "Given the numbers: {problem_description}, propose {n} possible next steps.\n"
            "Each step should combine **exactly two numbers** using one of the operations +, -, *, /,\n"
            "and show the result as a new number, along with the remaining numbers.\n"
            "Avoid invalid operations (e.g., division by zero).\n\n"
            "Format each step like this: \n"
            "a + b = c (left: c)\n\n"
            "→ where a and b are two numbers from the current list, c is the result,\n"
            "Possible next steps:\n"
        )
    return PromptTemplate.from_template(template)


class Game24PromptStrategy(BaseThoughtGenerationStrategy):
    """
    Game 24 strategy that generates expressions likely to evaluate to 24.
    """
    level: int = 0
    prompt: BasePromptTemplate = None
    tot_memory: Dict[Tuple[str, ...], List[str]] = Field(default_factory=dict)
    c: int = 5

    def __init__(self, level: int = 0, c: int =5, **kwargs):
        super().__init__(**kwargs)
        self.level = level
        self.prompt = get_game24_prompt(level)
        self.c = c
    def next_thought(
            self,
            problem_description: str,  # 就是 [3, 3, 8, 8]
            thoughts_path: Tuple[str, ...] = (),
            **kwargs: Any,
    ) -> str:
        if thoughts_path not in self.tot_memory or not self.tot_memory[thoughts_path]:
            # 生成新的一批 thought（候选表达式）
            thoughts_str = ' '.join(thoughts_path)
            new_thoughts = self.predict_and_parse(
                problem_description=str(problem_description),  # 转为 string
                #thoughts=thoughts_str,
                n=self.c,
                **kwargs,
            )
            if not new_thoughts:
                return ""
            if isinstance(new_thoughts, list):
                self.tot_memory[thoughts_path] = new_thoughts[::-1]  # 后进先出
            else:
                return ""

        return self.tot_memory[thoughts_path].pop()

    def predict_and_parse(
            self,
            problem_description: str,
            thoughts: Tuple[str, ...] = (),
            n: int = None,  # 要生成的候选解个数
            **kwargs: Any,
    ) -> List[str]:

        """
        调用 LLM 生成推理步骤，并尝试解析其输出。
        """
        if n is None:
            n = self.c
        # 格式化 prompt，并确保传递 n 作为参数
        prompt_text = self.prompt.format(problem_description=problem_description, thoughts=thoughts, n=n)
        try:
            # 调用 LLM 获取 raw output
            raw_output = self.llm.predict(prompt_text, **kwargs)

            # 打印原始输出，帮助调试
            # 使用 output_parser 来解析 LLM 输出
            parsed = self.output_parser.parse(raw_output)

            if isinstance(parsed, list):
                # 如果解析结果是列表，返回列表（多个候选解）
                return parsed[:n]  # 限制生成的候选解个数
            else:
                fallback = parse_game24_steps(raw_output)

                return fallback[:n]

        except Exception as e:
            # 出现异常时打印错误信息，并返回空列表
            print("[Error] Failed to parse:", e)
            return []


