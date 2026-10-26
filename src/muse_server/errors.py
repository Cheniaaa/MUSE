"""资源查找的明确缺失语义，与内部字典缺字段错误区分。"""


class ResourceNotFound(LookupError):
    """Run、预览、历史会话或已发布产物不存在。"""
