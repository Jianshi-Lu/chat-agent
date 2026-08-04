"""定义服务 Agent 使用的工具和 LangChain 检索适配。

``@tool`` 会把普通 Python 函数注册为 LangChain Agent 可选择调用的工具：
模型根据工具名称、描述和参数 schema 决定是否发起工具调用，LangChain
负责执行函数并把返回值回传给模型。两个占位工具只打印调用标识并返回占位文本；
``build_retriever_tool`` 则负责把原始 Chroma 检索结果适配为 LangChain Tool。
"""

from typing import Any

from chromadb.api.types import QueryResult
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, create_retriever_tool, tool
from pydantic import PrivateAttr

from ..retrieval import Retriever


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


def build_retriever_tool(
    retriever_instance: Retriever,
    *,
    top_k: int = 3,
) -> BaseTool:
    """把原始 Chroma Retriever 包装成 Agent 可注册的 ``retrieve`` Tool。"""

    class _ChromaRetrieverAdapter(BaseRetriever):
        """只在当前工厂内部把 Chroma 检索适配成 LangChain Retriever。"""

        _retriever: Retriever = PrivateAttr()
        _top_k: int = PrivateAttr(default=3)

        def __init__(
            self,
            retriever_instance: Retriever,
            *,
            top_k: int = 3,
        ) -> None:
            super().__init__()
            if top_k <= 0:
                raise ValueError("top_k 必须大于 0")
            self._retriever = retriever_instance
            self._top_k = top_k

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Any = None,
        ) -> list[Document]:
            """调用原始 Chroma 检索并转换为 LangChain Document。"""

            del run_manager
            result = self._retriever.retrieve(query, top_k=self._top_k)
            return self._to_documents(result)

        @staticmethod
        def _to_documents(result: QueryResult) -> list[Document]:
            """把 Chroma 批量查询结果转换成带来源信息的 Document。"""

            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            ids = result.get("ids") or []
            distances = result.get("distances") or []
            document_list = documents[0] if documents else []
            metadata_list = metadatas[0] if metadatas else []
            id_list = ids[0] if ids else []
            distance_list = distances[0] if distances else []

            converted: list[Document] = []
            for index, document in enumerate(document_list):
                raw_metadata = (
                    metadata_list[index] if index < len(metadata_list) else None
                )
                metadata = dict(raw_metadata or {})
                metadata["id"] = (
                    id_list[index] if index < len(id_list) else "未知"
                )
                metadata.setdefault("title", "未知")
                metadata.setdefault("date", "未知")
                metadata.setdefault("source", "未知")
                metadata["distance"] = (
                    distance_list[index] if index < len(distance_list) else "未知"
                )
                converted.append(Document(page_content=document, metadata=metadata))
            return converted

    document_prompt = PromptTemplate.from_template(
        "文档ID：{id}\n"
        "标题：{title}\n"
        "日期：{date}\n"
        "来源：{source}\n"
        "相似度距离：{distance}\n"
        "正文：\n{page_content}"
    )

    return create_retriever_tool(
        _ChromaRetrieverAdapter(retriever_instance, top_k=top_k),
        name="retrieve",
        description=(
            "检索政策知识库中的相关文本片段。涉及政策事实、政策定义、"
            "政策要求或政策来源的问题，应优先调用此工具。"
        ),
        document_prompt=document_prompt,
    )


__all__ = ["build_retriever_tool", "exact_paragraph_search", "semantic_search"]
