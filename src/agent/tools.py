"""定义对话 Agent 使用的占位工具。

当前工具已注册到 ``LangChainChatAgent``，只保留后续检索所需的函数形态并
打印调用标识；不连接 MySQL、ChromaDB 或其他外部服务。
"""

from langchain_core.tools import tool


@tool
def exact_paragraph_search(chapter_no: int, paragraph_no: int) -> str:
    """精确定位用户明确指定章节号和段落号的文档内容。

    只用于“第几章第几段”这类结构化位置查询，不用于根据语义查找相关内容。
    """

    print("模型调用工具：exact_paragraph_search")
    return f"占位结果：第{chapter_no}章第{paragraph_no}段"


@tool
def semantic_search(query: str) -> str:
    """根据用户问题检索语义相关内容。

    用于用户没有明确指定章节和段落、而是询问某个主题或概念的情况。
    """

    print("模型调用工具：semantic_search")
    return f"占位结果：语义检索 {query}"


__all__ = ["exact_paragraph_search", "semantic_search"]
