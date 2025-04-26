from langchain_experimental.tot.checker import ToTChecker
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from typing import List
import re
from pydantic import Field

llm = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",  # Azure 上你创建的模型部署名称
    api_key="4aa4a96857d34b759c149840af2d4641",
    api_version="2024-10-21",
    azure_endpoint="https://hkust.azure-api.net",  # 注意是 azure_endpoint 而不是 api_base
    temperature=0.7,
)


class Game24Checker(ToTChecker):
    scores: dict = Field(default_factory=lambda: {
        "sure": 20.0,
        "likely": 1.0,
        "impossible": 0.01
    })

    def input_keys(self) -> List[str]:
        return ["problem_description", "thoughts"]

    def output_keys(self) -> List[str]:
        return ["evaluation_result"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 调用父类构造函数
        self.scores = {
            "sure": 20.0,
            "likely": 1.0,
            "impossible": 0.01
        }

    def evaluate(self, x: str) -> float:
        """
        x: 当前路径的所有步骤文本
        y: 当前新思路（如一步计算或表达式）
        返回评分: sure=20.0, likely=1.0, impossible=0.01
        """
        prompt = self.build_value_prompt(x)

        response = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        #print(f"[Debug] Response: {response}")
        match = re.search(r"\b(sure|likely|impossible)\b", response.lower())
        if match:
            conclusion = match.group(1)  # 提取匹配的结论（"sure"、"likely"、"impossible"等）
            score = self.scores.get(conclusion, 0.01)  # 从 scores 字典中获取对应的分数，默认 0.01
            return score
        return self.scores.get(0.01)  # 默认返回最小分数

    def build_value_prompt(self, reasoning_text: str) -> str:
        return f'''Evaluate if given numbers can reach 24 (sure/likely/impossible). Use all the number only once.
                10 14
                10 + 14 = 24
                sure
                11 12
                11 + 12 = 23
                12 - 11 = 1
                11 * 12 = 132
                11 / 12 = 0.91
                impossible
                4 5 10
                4 + 5 + 10 = 9 + 10 = 18
                4 * 10 - 5 = 40 - 5 = 35
                (10 - 4) * 5 = 6 * 5 = 30
                sure
                4 9 11
                9 + 11 + 4 = 20 + 4 = 24
                sure
                5 7 8
                5 + 7 + 8 = 12 + 8 = 20
                (8 - 5) * 7 = 3 * 7 = 21
                I cannot obtain 24 now, but numbers are within a reasonable range
                likely
                5 6 7
                5 + 6 + 7 = 18
                (6 - 5) * 7 = 1 * 7 = 7
                I cannot obtain 24 now, but numbers are within a reasonable range
                likely
                10 10 11
                10 + 10 + 11 = 31
                (11 - 10) * 10 = 10
                10 10 10 are all too big
                impossible
                1 3 3
                1 * 3 * 3 = 9
                (1 + 3) * 3 = 12
                1 3 3 are all too small
                impossible
                {reasoning_text}
                '''.strip()

    def extract_numbers_from_text(self, text: str) -> List[int]:
        import re
        return sorted([int(x) for x in re.findall(r'\b\d+\b', text)])
