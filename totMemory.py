from thought import Thought
from langchain_experimental.tot.memory import ToTDFSMemory
class CustomToTDFSMemory(ToTDFSMemory):
    def __init__(self):
        super().__init__()
        self.stack = []  # 栈，用来存储当前的思维节点

    def store(self, node: Thought) -> None:
        """
        存储一个新的思维节点到栈中
        """
        if len(self.stack) > 0:
            # 如果栈不为空，设置当前栈顶节点的子节点
            self.stack[-1].children.add(node)
        self.stack.append(node)  # 将当前节点推入栈中

    def pop(self) -> None:
        """
        弹出栈顶节点
        """
        if self.stack:
            self.stack.pop()

    def get_current_node(self) -> Thought:
        """
        获取栈顶节点
        """
        return self.stack[-1] if self.stack else None

    def get_all_thoughts(self) -> list[Thought]:
        """
        返回所有存储的思维节点
        """
        return self.stack

    @property
    def level(self) -> int:
        """
        计算并返回当前的思维层级（即栈的深度）
        """
        return len(self.stack)