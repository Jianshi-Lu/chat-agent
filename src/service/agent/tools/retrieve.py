"""定义 Agent 的唯一向量检索 Tool 及其运行时依赖。"""

from dataclasses import dataclass
from typing import Protocol

from langchain.tools import ToolRuntime
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field, field_validator

from src.errors import AppException, KNOWLEDGE_RETRIEVAL_FAILED


class QueryEmbedder(Protocol):
    """向量检索 Tool 依赖的最小查询编码接口。"""

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """一次返回全部查询的向量。"""


class RetrieveInput(BaseModel):
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


@dataclass(frozen=True, slots=True)
class RetrievalToolContext:
    """向 Agent Tool 注入当前 Runtime 持有的向量检索依赖。"""

    vector_store: VectorStore
    embeddings: QueryEmbedder
    top_k: int


@tool(args_schema=RetrieveInput)
def retrieve(
    queries: list[str],
    runtime: ToolRuntime[RetrievalToolContext],
) -> str:
    """向量化查询并检索知识库，返回每个查询对应的知识片段。

    涉及政策事实、定义、要求或来源时调用；若用户一轮提出多个问题，
    必须把全部子问题放入同一个 queries 数组，只调用一次本工具。

    本函数是完整的向量检索边界，内部依次完成：查询校验、批量查询向量化、
    向量数量校验、Chroma 相似度检索和结果格式化。

    依赖传导链：``Runtime.retrieval_context`` → ``RetrievalToolContext`` →
    ``runtime.context.embeddings`` / ``runtime.context.vector_store`` → 本函数。
    ``runtime`` 是 LangChain 在工具执行时自动注入的运行时参数，
    不会暴露给模型作为工具入参。
    """

    context = runtime.context
    if isinstance(queries, (str, bytes)) or not queries:
        raise _error(
            stage="validate_queries",
            message="批量检索至少需要一个查询",
        )

    normalized_queries: list[str] = []
    for index, query in enumerate(queries):
        if not isinstance(query, str) or not query.strip():
            raise _error(
                stage="validate_queries",
                message="批量检索查询不能为空",
                details={"query_index": index},
            )
        normalized_queries.append(query.strip())

    if context.top_k <= 0:
        raise _error(
            stage="validate_config",
            message="向量检索 top_k 必须大于 0",
            details={"top_k": context.top_k},
        )

    try:
        # 将请求向量化
        vectors = context.embeddings.embed_queries(normalized_queries)
    except AppException:
        raise
    except Exception as exc:
        raise _error(
            stage="embed_queries",
            message="批量查询向量生成失败",
            details={"query_count": len(normalized_queries)},
        ) from exc

    if len(vectors) != len(normalized_queries):
        raise _error(
            stage="validate_embeddings",
            message="批量查询向量数量与查询数量不一致",
            details={
                "query_count": len(normalized_queries),
                "embedding_count": len(vectors),
            },
        )

    blocks: list[str] = []
    for index, (query, vector) in enumerate(zip(normalized_queries, vectors)):
        try:
            # 向量检索数据库
            documents = context.vector_store.similarity_search_by_vector(
                vector,
                k=context.top_k,
            )
        except Exception as exc:
            raise _error(
                stage="search_vector_store",
                message="知识库向量检索失败",
                details={"query_index": index, "query": query},
            ) from exc

        lines = [f"查询 {index + 1}：{query}"]
        if not documents:
            lines.append("未检索到相关知识片段。")
        for document_index, document in enumerate(documents, start=1):
            document_id = document.id or document.metadata.get("id", "未知")
            source = document.metadata.get("source", "未知")
            lines.append(
                f"结果 {document_index}\n"
                f"文档ID：{document_id}\n"
                f"来源：{source}\n"
                f"正文：\n{document.page_content}"
            )
        blocks.append("\n".join(lines))

    print(f"模型调用工具：retrieve（{len(queries)} 个查询）", flush=True)
    return "\n\n".join(blocks)


def _error(
    *,
    stage: str,
    message: str,
    details: dict[str, object] | None = None,
) -> AppException:
    return AppException(
        KNOWLEDGE_RETRIEVAL_FAILED,
        module="service.agent.tools.retrieve",
        operation="retrieve",
        stage=stage,
        message=message,
        details=details,
    )


__all__ = [
    "QueryEmbedder",
    "RetrieveInput",
    "RetrievalToolContext",
    "retrieve",
]
