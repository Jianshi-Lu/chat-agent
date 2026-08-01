"""定义服务 Agent 使用的工具占位实现。

``@tool`` 会把普通 Python 函数注册为 LangChain Agent 可选择调用的工具：
模型根据工具名称、描述和参数 schema 决定是否发起工具调用，LangChain
负责执行函数并把返回值回传给模型。当前函数只打印调用标识并返回占位文本，
不连接 ChromaDB 或其他外部服务。
"""

from langchain_core.tools import tool


@tool
def exact_paragraph_search(chapter_no: int, paragraph_no: int) -> str:
    """精确定位用户明确指定章节号和段落号的文档内容。"""

    print("模型调用工具：exact_paragraph_search", flush=True)
    return f"占位结果：第{chapter_no}章第{paragraph_no}段"


@tool
def semantic_search(query: str) -> str:
    """根据用户问题检索语义相关内容。"""

    print("模型调用工具：semantic_search", flush=True)
    return f"占位结果：语义检索 {query}"


__all__ = ["exact_paragraph_search", "semantic_search"]
