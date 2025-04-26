from typing import Optional
from langchain_experimental.tot.thought import ThoughtValidity

class Thought:
    text: str
    score: Optional[float] = None  # `score` 是可选的
    level: int = 0  # 默认为 0
    validity: Optional[ThoughtValidity] = ThoughtValidity.VALID_FINAL
    children: set = None  # 默认为空集合

    def __init__(self, text: str, score: float, level: int = 0):
        self.text = text
        self.score = score
        self.level = level  # 每个节点都有自己的层级
        self.children = set()

    def __post_init__(self):
        if self.children is None:
            self.children = set()

    def __repr__(self):
        if self.score is not None:
            return f"Thought(score={self.score:.2f}): {self.text}"
        else:
            return f"Thought(level={self.level}): {self.text}"
