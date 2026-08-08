"""创建 Chroma VectorStore 并负责 Document 持久化。"""

from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from configuration import get_ingestion_settings
from src.errors import AppException, KNOWLEDGE_INGESTION_FAILED


def create_vector_store(
    *,
    embeddings: Embeddings,
    chroma_path: str | Path,
    collection_name: str | None = None,
) -> Chroma:
    """创建连接持久化 Chroma collection 的官方 ``Chroma`` 实例。"""

    if collection_name is None:
        collection_name = get_ingestion_settings().collection_name

    try:
        path = Path(chroma_path)
        path.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(path),
            collection_metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        raise _error(
            stage="initialize_vector_store",
            message="持久化 Chroma collection 创建失败",
            details={"chroma_path": str(chroma_path)},
        ) from exc


def persist_documents(
    vector_store: Chroma,
    documents: Sequence[Document],
) -> list[str]:
    """调用官方 ``Chroma.add_documents`` 持久化已准备好的 Document。"""

    ids = [document.id for document in documents]
    if not documents or any(document_id is None for document_id in ids):
        raise _error(
            stage="validate_documents",
            message="写入 Chroma 的 Document 必须非空且包含 id",
        )

    try:
        return vector_store.add_documents(
            list(documents),
            ids=[document_id for document_id in ids if document_id is not None],
        )
    except Exception as exc:
        raise _error(
            stage="store_documents",
            message="文档向量写入持久化 Chroma 失败",
        ) from exc


def delete_document_chunks(
    vector_store: Chroma,
    document_uuid: str,
) -> int:
    """按稳定 document_uuid 删除一个知识文件产生的全部 chunk。"""

    if not isinstance(document_uuid, str) or not document_uuid.strip():
        raise _error(
            stage="validate_document_uuid",
            message="删除向量时 document_uuid 不能为空",
        )

    try:
        result = vector_store.get(
            where={"document_uuid": document_uuid.strip()},
            include=[],
        )
        ids = list(result.get("ids") or [])
        if ids:
            vector_store.delete(ids=ids)
        return len(ids)
    except Exception as exc:
        raise _error(
            stage="delete_document_chunks",
            message="知识文件向量删除失败",
            details={"document_uuid": document_uuid},
        ) from exc


def count_document_chunks(
    vector_store: Chroma,
    document_uuid: str,
) -> int:
    """查询一个 document_uuid 当前在 Chroma 中的 chunk 数量（不存在为 0）。"""

    if not isinstance(document_uuid, str) or not document_uuid.strip():
        raise _error(
            stage="validate_document_uuid",
            message="查询向量状态时 document_uuid 不能为空",
        )

    try:
        result = vector_store.get(
            where={"document_uuid": document_uuid.strip()},
            include=[],
        )
        return len(result.get("ids") or [])
    except Exception as exc:
        raise _error(
            stage="count_document_chunks",
            message="知识文件向量状态查询失败",
            details={"document_uuid": document_uuid},
        ) from exc


def _error(
    *,
    stage: str,
    message: str,
    details: dict[str, object] | None = None,
) -> AppException:
    """构造 Persistence 模块的结构化异常。"""

    return AppException(
        KNOWLEDGE_INGESTION_FAILED,
        module="service.ingestion.persistence",
        operation="persist_documents",
        stage=stage,
        message=message,
        details=details,
    )


__all__ = [
    "count_document_chunks",
    "create_vector_store",
    "delete_document_chunks",
    "persist_documents",
]
