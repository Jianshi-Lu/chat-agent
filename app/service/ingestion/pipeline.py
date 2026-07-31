"""实现从 Markdown 文件到内存 Chroma collection 的最小链路。"""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.errors import AppException, KNOWLEDGE_INGESTION_FAILED


DEFAULT_EMBEDDING_MODEL_PATH = Path(
    "/Users/lujianshi/Desktop/models/embedding/Qwen:Qwen3-Embedding-8B"
)


@dataclass(frozen=True, slots=True)
class TextChunk:
    """保存一个可写入向量库的文本块及其来源信息。"""

    id: str
    text: str
    metadata: dict[str, str | int]


class IngestionService:
    """执行分块、Embedding、内存 Chroma 写入和相似度查询。"""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_EMBEDDING_MODEL_PATH,
        device: str | None = None,
    ) -> None:
        """加载本地 Embedding 模型并创建临时 Chroma collection。"""

        model_path = Path(model_path)
        if not model_path.is_dir():
            raise self._error(
                stage="initialize_model",
                message="本地 Embedding 模型目录不存在",
            )

        try:
            self.model = SentenceTransformer(
                str(model_path),
                device=device,
                local_files_only=True,
            )
        except Exception as exc:
            raise self._error(
                stage="initialize_model",
                message="本地 Embedding 模型加载失败",
            ) from exc

        try:
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(
                name="policy_chunks",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise self._error(
                stage="initialize_vector_store",
                message="内存 Chroma collection 创建失败",
            ) from exc

    def ingest(
        self,
        path: str | Path,
        *,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ) -> list[TextChunk]:
        """读取一个 Markdown 文件，分块并写入内存 Chroma。"""

        file_path = Path(path)
        chunks = self._chunk_content(
            self._read_markdown(file_path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise self._error(
                stage="validate_content",
                message="Markdown 正文不能为空",
            )

        try:
            embeddings = self.model.encode(
                [chunk.text for chunk in chunks],
                normalize_embeddings=True,
                batch_size=1,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise self._error(
                stage="encode_documents",
                message="政策文档向量化失败",
            ) from exc

        try:
            self.collection.add(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {**chunk.metadata, "source": str(file_path)}
                    for chunk in chunks
                ],
                embeddings=embeddings.tolist(),
            )
        except Exception as exc:
            raise self._error(
                stage="store_chunks",
                message="文档向量写入内存 Chroma 失败",
            ) from exc
        return chunks

    def search(self, query: str, *, top_k: int = 3) -> dict:
        """将查询向量化，并从内存 Chroma 返回最相近的文本块。"""

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
        if self.collection.count() == 0:
            raise self._error(
                stage="query_collection",
                message="内存 Chroma collection 尚未写入文档",
            )

        try:
            query_embedding = self.model.encode(
                query,
                prompt_name="query",
                normalize_embeddings=True,
                batch_size=1,
                show_progress_bar=False,
            )
            return self.collection.query(
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
    def _read_markdown(path: str | Path) -> str:
        """读取 UTF-8 Markdown 文件。"""

        try:
            return Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise IngestionService._error(
                stage="read_markdown",
                message="Markdown 文件读取失败",
            ) from exc

    @staticmethod
    def _chunk_content(
        content: str,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[TextChunk]:
        """按字符窗口切分 Markdown 正文。"""

        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise IngestionService._error(
                stage="validate_chunks",
                message="chunk_size 必须大于 0，chunk_overlap 必须小于 chunk_size",
                details={
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                },
            )

        chunks: list[TextChunk] = []
        step = chunk_size - chunk_overlap
        for index, start in enumerate(range(0, len(content), step)):
            chunks.append(
                TextChunk(
                    id=f"chunk-{index}",
                    text=content[start : start + chunk_size],
                    metadata={"chunk_index": index},
                )
            )
        return chunks

    @staticmethod
    def _error(
        *,
        stage: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> AppException:
        """构造 ingestion 模块统一使用的应用异常。"""

        return AppException(
            KNOWLEDGE_INGESTION_FAILED,
            module="service.ingestion",
            operation="ingest",
            stage=stage,
            message=message,
            details=details,
        )
