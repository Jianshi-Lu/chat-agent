"""定义服务 Agent 使用的工具和批量检索 Tool 工厂。"""

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import BaseTool, StructuredTool, tool

from src.service.retrieval import BatchRetriever, QueryRetrievalResult


class RetrieveManyInput(BaseModel):
    """要求模型把同一轮的全部检索子问题放入一个列表。"""

    queries: list[str] = Field(
        min_length=1,
        description="需要同时检索的全部独立问题；同一轮只提交一个 queries 列表。",
    )

    @field_validator("queries")
    @classmethod
    def normalize_queries(cls, queries: list[str]) -> list[str]:
        normalized = [query.strip() for query in queries]
        if any(not query for query in normalized):
            raise ValueError("queries 中不能包含空查询")
        return normalized


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


def _format_results(results: list[QueryRetrievalResult]) -> str:
    blocks: list[str] = []
    for query_index, result in enumerate(results, start=1):
        lines = [f"查询 {query_index}：{result.query}"]
        if not result.documents:
            lines.append("未检索到相关知识片段。")
        for document_index, document in enumerate(result.documents, start=1):
            document_id = document.id or document.metadata.get("id", "未知")
            source = document.metadata.get("source", "未知")
            lines.append(
                f"结果 {document_index}\n"
                f"文档ID：{document_id}\n"
                f"来源：{source}\n"
                f"正文：\n{document.page_content}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_retriever_tool(retriever_instance: BatchRetriever) -> BaseTool:
    """把批量 Retriever 包装成结构化 ``retrieve_many`` Tool。"""

    def retrieve_many(queries: list[str]) -> str:
        print(f"模型调用工具：retrieve_many（{len(queries)} 个查询）", flush=True)
        return _format_results(retriever_instance.retrieve_many(queries))

    return StructuredTool.from_function(
        func=retrieve_many,
        name="retrieve_many",
        description=(
            "批量检索政策知识库。涉及政策事实、定义、要求或来源时调用；"
            "若用户一轮提出多个问题，必须把全部子问题放入同一个 queries 数组，"
            "只调用一次本工具，不要为每个子问题分别发起工具调用。"
        ),
        args_schema=RetrieveManyInput,
    )


__all__ = [
    "RetrieveManyInput",
    "build_retriever_tool",
    "exact_paragraph_search",
    "semantic_search",
]
