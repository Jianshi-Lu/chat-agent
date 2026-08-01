"""实现基于 Embedding 和内存 Chroma 的最小向量检索器。"""

from typing import Any

from src.errors import AppException, KNOWLEDGE_RETRIEVAL_FAILED

from ..ingestion import IngestionService


class VectorRetriever:
    """复用入库服务中的模型和 collection 执行相似度检索。"""

    def __init__(self, ingestion: IngestionService) -> None:
        """接收已经完成模型初始化的入库服务，不重复创建向量库。"""

        self.ingestion = ingestion

    def retrieve(self, query: str, *, top_k: int = 3) -> dict[str, Any]:
        """将问题向量化，并返回 Chroma 中最相关的文本片段。"""

        if not isinstance(query, str) or not query.strip():
            raise self._error(
                stage="validate_query",
                message="查询文本不能为空",
            )
        if top_k <= 0:
            raise self._error(
                stage="validate_query",
                message="top_k 必须大于 0",
                details={"top_k": top_k},
            )
        if self.ingestion.collection.count() == 0:
            raise self._error(
                stage="query_collection",
                message="内存 Chroma collection 尚未写入文档",
            )

        try:
            query_embedding = self.ingestion.model.encode(
                query,
                prompt_name="query",
                normalize_embeddings=True,
                batch_size=1,
                show_progress_bar=False,
            )
            return self.ingestion.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise self._error(
                stage="query_collection",
                message="内存 Chroma 查询失败",
            ) from exc

    @staticmethod
    def _error(
        *,
        stage: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> AppException:
        """构造 retrieval 模块统一使用的应用异常。"""

        return AppException(
            KNOWLEDGE_RETRIEVAL_FAILED,
            module="service.retrieval",
            operation="retrieve",
            stage=stage,
            message=message,
            details=details,
        )


__all__ = ["VectorRetriever"]
