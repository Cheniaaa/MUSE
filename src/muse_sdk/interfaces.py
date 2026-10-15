"""算法的显式 episode reset 契约；不导入 tensor 实现。"""


class Algorithm:
    def reset_worlds(self, mask):
        """清除 bool[B] 选中 world 的 episode 状态；无状态算法默认无需操作。"""
        return None
