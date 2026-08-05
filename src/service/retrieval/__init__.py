"""知识检索领域接口。"""

from .batch import BatchQueryEmbedder, BatchRetriever, QueryRetrievalResult

__all__ = ["BatchQueryEmbedder", "BatchRetriever", "QueryRetrievalResult"]
