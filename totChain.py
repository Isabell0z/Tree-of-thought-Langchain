from checker import Game24Checker
from langchain.chains.base import Chain
from langchain.base_language import BaseLanguageModel
from langchain_experimental.tot.controller import ToTController
from typing import Any, Dict, List, Optional, Type
from langchain_experimental.tot.thought_generation import BaseThoughtGenerationStrategy
from promptStrategy import Game24PromptStrategy
from thought import Thought
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_experimental.tot.thought import ThoughtValidity
from textwrap import indent
import re
from totMemory import CustomToTDFSMemory
from langchain_experimental.tot.base import ToTChain


def extract_left_numbers_from_thought(thought: str) -> str:
    match = re.search(r'\(left:\s*([^\)]+)\)', thought)
    if match:
        return match.group(1).strip()
    return ""


def extract_number_after_equal(thought: str) -> str:
    match = re.search(r'[=≈]\s*(-?\d+(?:\.\d+)?)', thought)
    if match:
        return match.group(1)
    return ""  # 如果没有找到匹配的数字，返回空字符串


class TotChain(Chain):
    """
    Chain implementing the Tree of Thought (ToT).
    """

    llm: BaseLanguageModel
    """
    Language model to use. It must be set to produce different variations for
    the same prompt.
    """
    checker: Game24Checker = Game24Checker()
    """ToT Checker to use."""
    output_key: str = "response"  #: :meta private:
    k: int = 5
    """The maximum number of conversation rounds"""
    c: int = 3
    """The number of children to explore at each node"""
    tot_memory: CustomToTDFSMemory = CustomToTDFSMemory()
    tot_controller: ToTController = ToTController()
    tot_strategy_class: Type[BaseThoughtGenerationStrategy] = Game24PromptStrategy
    verbose_llm: bool = False
    acc: int = 0
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, **kwargs: Any) -> ToTChain:
        """
        Create a ToTChain from a language model.

        :param llm: The language model to use.
        :param kwargs: Additional arguments to pass to the ToTChain constructor.
        """
        return cls(llm=llm, **kwargs)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tot_controller.c = self.c

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return ["problem_description"]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return ["output"]

    def log_thought(
            self,
            thought: Thought,
            level: int,
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> None:
        if run_manager:
            if thought.score is not None:
                color = "cyan"  # 用不同颜色区分
                text = indent(f"Thought: {thought.text} | Score: {thought.score:.2f}\n", prefix="    " * level)
            else:
                colors = {
                    ThoughtValidity.VALID_FINAL: "green",
                    ThoughtValidity.VALID_INTERMEDIATE: "yellow",
                    ThoughtValidity.INVALID: "red",
                }
                color = colors.get(thought.validity, "gray")  # 如果validity未知，使用灰色
                text = indent(f"Thought: {thought.text}\n", prefix="    " * level)

            run_manager.on_text(text=text, color=color, verbose=self.verbose)

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        with open("log_tot_1.txt", "a") as f:  # "a" 模式表示追加写入
            _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
            if run_manager:
                run_manager.on_text(text="Starting the ToT solve procedure.\n")

            problem_description = inputs["problem_description"]
            thoughts_path: tuple[str, ...] = ()
            # thought_generator = Game24PromptStrategy(  # type: ignore[call-arg]
            #    llm=self.llm, c=self.c, verbose=self.verbose_llm
            # )
            s = ",".join(map(str, problem_description))
            s = "" + "(left:" + s + ")"
            current_paths = [{"path": (s,), "score": 20}]
            for level in range(3):  # 控制层数
                thought_generator = Game24PromptStrategy(  # type: ignore[call-arg]
                    llm=self.llm, c=self.c * (3 - level), verbose=self.verbose_llm, level=level
                )
                all_new_paths = []
                f.write(f"Level: {level}\n")
                print(f"[Debug] Level: {level}")
                for path in current_paths:
                    lst = list(map(float, re.split(r"[, ]+", extract_left_numbers_from_thought(path['path'][-1]).strip())))
                    lsts = ' '.join(map(str, lst))
                    for i in range(self.c):
                        scored_thoughts = []

                        thought_texts = thought_generator.next_thought(
                            lsts, path['path'], callbacks=_run_manager.get_child()
                        )

                        # thoughts_path += (thought_texts,)

                        if isinstance(thought_texts, str):
                            thought_texts = [thought_texts]

                        for thought_text in thought_texts:
                            # 构造 checker 的打分输入
                            if thought_text != '':
                                x = extract_left_numbers_from_thought(thought_text)
                                # 使用新的 value checker 给 thought 打分
                                score = self.checker.evaluate(x)
                                thought = Thought(text=thought_text, score=score, level=level)
                                scored_thoughts.append(thought)

                        # top_thoughts = sorted(scored_thoughts, key=lambda t: t.score, reverse=True)[:5]
                        # print("top path:",top_thoughts)

                        # for thought in top_thoughts:
                        #    self.tot_memory.store(thought)
                        #    self.log_thought(thought, level, run_manager)
                        # 控制器决定下一步的思路路径（可基于得分排序）
                        for t in scored_thoughts:
                            new_path = path['path'] + (t.text,)
                            all_new_paths.append({"path": new_path, "score": t.score})
                if level == 2:
                    for t in all_new_paths:
                        value = eval(extract_number_after_equal(t['path'][-1]))
                        if abs(value - 24) < 1e-2:
                            f.write(f"correct path: {t['path']}\n")
                            print(t['path'])
                            self.acc = 1
                            return {"output": str(self.acc)}
                            #return {"output": " -> ".join(t['path'])}
                current_paths = sorted(all_new_paths, key=lambda x: x["score"], reverse=True)[:1]
                print("current path:", current_paths)
                f.write(f"current path: {current_paths}\n")
        return {"output": str(self.acc)}

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Async not implemented yet")

    @property
    def _chain_type(self) -> str:
        return "tot"
