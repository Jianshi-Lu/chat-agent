"""一次编码多个查询并分别执行向量检索。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.errors import AppException, KNOWLEDGE_RETRIEVAL_FAILED


class BatchQueryEmbedder(Protocol):
    """检索层依赖的最小批量查询编码契约。"""

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """一次返回全部查询的向量。"""


@dataclass(frozen=True, slots=True)
class QueryRetrievalResult:
    """保存一个查询及其对应的知识片段。"""

    query: str
    documents: tuple[Document, ...]


class BatchRetriever:
    """将 N 个查询合并编码，再按向量逐个查询 VectorStore。"""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embeddings: BatchQueryEmbedder,
        top_k: int,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.top_k = top_k

    def retrieve_many(
        self,
        queries: Sequence[str],
    ) -> list[QueryRetrievalResult]:
        """只调用一次批量 Embedding，并保留每个查询的独立检索结果。"""

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

        try:
            vectors = self.embeddings.embed_queries(normalized_queries)
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

        results: list[QueryRetrievalResult] = []
        for index, (query, vector) in enumerate(zip(normalized_queries, vectors)):
            try:
                documents = self.vector_store.similarity_search_by_vector(
                    vector,
                    k=self.top_k,
                )
            except Exception as exc:
                raise _error(
                    stage="search_vector_store",
                    message="知识库向量检索失败",
                    details={"query_index": index, "query": query},
                ) from exc
            results.append(
                QueryRetrievalResult(
                    query=query,
                    documents=tuple(documents),
                )
            )
        return results


def _error(
    *,
    stage: str,
    message: str,
    details: dict[str, object] | None = None,
) -> AppException:
    return AppException(
        KNOWLEDGE_RETRIEVAL_FAILED,
        module="service.retrieval.batch",
        operation="retrieve_many",
        stage=stage,
        message=message,
        details=details,
    )


__all__ = ["BatchQueryEmbedder", "BatchRetriever", "QueryRetrievalResult"]
