"""组装源资料解析、切分、向量化和 Chroma 持久化流程。"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.errors import AppException, KNOWLEDGE_INGESTION_FAILED

from .embedding import EmbeddingRuntime, create_model
from .models import KnowledgeDocument
from .parser import parse_documents
from .persistence import (
    create_vector_store,
    delete_document_chunks,
    persist_documents,
)
from .splitter import create_splitter


class IngestionPipeline:
    """持有官方组件并按顺序组装一次知识入库任务。"""

    def __init__(
        self,
        *,
        model_path: str | Path,
        chroma_path: str | Path,
        collection_name: str | None = None,
        embedding_batch_size: int | None = None,
        query_embedding_batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        """创建 Embeddings 和 Chroma；不在初始化时读取或写入源文件。"""

        self.embeddings: EmbeddingRuntime = create_model(
            model_path,
            device=device,
            embedding_batch_size=embedding_batch_size,
            query_embedding_batch_size=query_embedding_batch_size,
        )
        self.vector_store: Chroma = create_vector_store(
            embeddings=self.embeddings,
            chroma_path=chroma_path,
            collection_name=collection_name,
        )

    def run(
        self,
        source_path: str | Path,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Document]:
        """按解析 → 按格式切分 → 生成 ID → 向量化 → 入库顺序执行任务。

        切分参数未传入时由 ``configuration.config`` 提供默认值。
        """

        source_documents = parse_documents(source_path)
        source_format = source_documents[0].metadata.get("format") if source_documents else None
        splitter = (
            create_splitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if source_format == "md"
            else None
        )

        documents: list[Document] = []
        source_name = Path(source_path).stem
        for document_index, source_document in enumerate(source_documents):
            chunks = (
                splitter.split_documents([source_document])
                if splitter is not None
                else [source_document]
            )
            for index, document in enumerate(chunks):
                chunk_id = f"{source_name}-document-{document_index}-chunk-{index}"
                document.id = chunk_id
                document.metadata.update(
                    {
                        "id": chunk_id,
                        "chunk_id": chunk_id,
                        "document_index": document_index,
                        "chunk_index": index,
                    }
                )
            documents.extend(chunks)

        if not documents:
            raise AppException(
                KNOWLEDGE_INGESTION_FAILED,
                module="service.ingestion.pipeline",
                operation="run",
                stage="validate_content",
                message="源文件中没有可入库的文档",
            )

        persist_documents(self.vector_store, documents)
        return documents

    def vectorize_document(
        self,
        source: KnowledgeDocument,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Document]:
        """切分并向量化一个带稳定 document_uuid 的 Markdown 文件。"""

        if not isinstance(source, KnowledgeDocument):
            raise AppException(
                KNOWLEDGE_INGESTION_FAILED,
                module="service.ingestion.pipeline",
                operation="vectorize_document",
                stage="validate_document",
                message="待向量化知识文件不合法",
            )

        source_document = Document(
            page_content=source.content,
            metadata=source.chroma_metadata(),
        )
        splitter = create_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents([source_document])
        if not chunks:
            raise AppException(
                KNOWLEDGE_INGESTION_FAILED,
                module="service.ingestion.pipeline",
                operation="vectorize_document",
                stage="validate_content",
                message="知识文件中没有可向量化的内容",
                details={"document_uuid": source.document_uuid},
            )

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{source.document_uuid}:chunk:{chunk_index}"
            chunk.id = chunk_id
            chunk.metadata.update(
                {
                    **source.chroma_metadata(),
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "document_index": 0,
                    "chunk_index": chunk_index,
                }
            )

        # 同一 document_uuid 重试时采用替换语义，避免生成重复 chunk。
        delete_document_chunks(self.vector_store, source.document_uuid)
        try:
            persist_documents(self.vector_store, chunks)
        except Exception:
            # add_documents 失败时可能已经写入部分数据，按 UUID 幂等清理。
            try:
                delete_document_chunks(self.vector_store, source.document_uuid)
            except AppException:
                pass
            raise
        return chunks

    def delete_document(self, document_uuid: str) -> int:
        """幂等删除一个知识文件产生的全部向量 chunk。"""

        return delete_document_chunks(self.vector_store, document_uuid)

    def close(self) -> None:
        """释放 Pipeline 创建的模型和 Chroma 客户端，但保留磁盘数据。"""

        vector_store = getattr(self, "vector_store", None)
        if vector_store is not None:
            try:
                vector_store._client.close()
                vector_store._client.clear_system_cache()
            except Exception:
                pass

        embeddings = getattr(self, "embeddings", None)
        close_embeddings = getattr(embeddings, "close", None)
        if callable(close_embeddings):
            try:
                close_embeddings()
            except Exception:
                pass

        self.vector_store = None
        self.embeddings = None


__all__ = ["IngestionPipeline"]
