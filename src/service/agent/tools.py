"""定义服务 Agent 使用的工具和 LangChain 检索工具工厂。

``@tool`` 会把普通 Python 函数注册为 LangChain Agent 可选择调用的工具：模型
根据工具名称、描述和参数 schema 决定是否发起工具调用，LangChain 负责执行函数
并把返回值回传给模型。两个占位工具保留现状；真实政策检索直接接收 LangChain
``BaseRetriever``，由 ``create_retriever_tool`` 完成 Tool 封装。
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, create_retriever_tool, tool


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


def build_retriever_tool(retriever_instance: BaseRetriever) -> BaseTool:
    """把 LangChain Retriever 包装成 Agent 可注册的 ``retrieve`` Tool。"""

    document_prompt = PromptTemplate.from_template(
        "文档ID：{id}\n"
        "来源：{source}\n"
        "正文：\n{page_content}"
    )

    return create_retriever_tool(
        retriever_instance,
        name="retrieve",
        description=(
            "检索政策知识库中的相关文本片段。涉及政策事实、政策定义、"
            "政策要求或政策来源的问题，应优先调用此工具。"
        ),
        document_prompt=document_prompt,
    )


__all__ = ["build_retriever_tool", "exact_paragraph_search", "semantic_search"]
