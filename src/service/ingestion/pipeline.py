"""实现从 ``data/policies.json`` 到持久化 Chroma collection 的最小链路。

处理链路为：政策 JSON → 读取政策正文 → 文本分块 → Embedding 向量化 →
写入指定目录中的 Chroma collection。首次启动会执行向量化，后续由 Runtime
检测已有 collection 并复用数据；服务退出时由 ``close`` 释放当前进程资源，
不会删除磁盘上的向量数据。
"""

import gc
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from src.errors import AppException, KNOWLEDGE_INGESTION_FAILED


DEFAULT_EMBEDDING_MODEL_PATH = Path(
    "/Users/lujianshi/Desktop/models/embedding/Qwen:Qwen3-Embedding-8B"
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICIES_PATH = PROJECT_ROOT / "data" / "policies.json"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"


@dataclass(frozen=True, slots=True)
class TextChunk:
    """保存一个可写入向量库的文本块及其来源信息。"""

    id: str
    text: str
    metadata: dict[str, str | int]


class IngestionService:
    """执行政策 JSON 分块、Embedding 和持久化 Chroma 写入。"""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_EMBEDDING_MODEL_PATH,
        chroma_path: str | Path = DEFAULT_CHROMA_PATH,
        device: str | None = None,
    ) -> None:
        """加载本地 Embedding 模型并打开指定目录中的 Chroma collection。"""
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise self._error(
                stage="initialize_model",
                message="本地 Embedding 模型目录不存在",
            )
        missing_data_files = [
            file_path.name
            for file_path in model_path.glob("*.safetensors")
            if file_path.stat().st_size == 0 or file_path.stat().st_blocks == 0
        ]
        if missing_data_files:
            raise self._error(
                stage="initialize_model",
                message=(
                    "本地 Embedding 模型权重分片没有实际数据，"
                    f"请重新下载：{', '.join(missing_data_files)}"
                ),
                details={"model_path": str(model_path)},
            )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        print(f"Embedding 推理设备：{device}", flush=True)

        try:
            self.embedding_model = SentenceTransformer(
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
            self.chroma_path = Path(chroma_path)
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.policy_chunk_collection = self.chroma_client.get_or_create_collection(
                name="policy_chunks",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise self._error(
                stage="initialize_vector_store",
                message="持久化 Chroma collection 创建失败",
                details={"chroma_path": str(chroma_path)},
            ) from exc

    def ingest(
        self,
        path: str | Path = DEFAULT_POLICIES_PATH,
        *,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ) -> list[TextChunk]:
        """读取政策 JSON，分块并写入指定目录中的 Chroma。"""

        file_path = Path(path)
        policies = self._read_policies(file_path)
        chunks: list[TextChunk] = []
        for policy in policies:
            content = (
                f"# {policy['title']}\n\n"
                f"类型：{policy['type']}\n"
                f"日期：{policy['date']}\n"
                f"来源：{policy['source']}\n\n"
                f"{policy['content']}"
            )
            policy_chunks = self._chunk_content(
                content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks.extend(
                TextChunk(
                    id=f"{policy['id']}-{chunk.id}",
                    text=chunk.text,
                    metadata={
                        **chunk.metadata,
                        "policy_id": policy["id"],
                        "title": policy["title"],
                        "type": policy["type"],
                        "date": policy["date"],
                        "source": policy["source"],
                        "file_path": str(file_path),
                    },
                )
                for chunk in policy_chunks
            )
        if not chunks:
            raise self._error(
                stage="validate_content",
                message="政策 JSON 中没有可入库的正文",
            )

        try:
            embeddings = self.embedding_model.encode(
                [chunk.text for chunk in chunks],
                normalize_embeddings=True,
                batch_size=4,
                show_progress_bar=True,
            )
        except Exception as exc:
            raise self._error(
                stage="encode_documents",
                message="政策文档向量化失败",
            ) from exc

        try:
            self.policy_chunk_collection.add(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                embeddings=embeddings.tolist(),
            )
        except Exception as exc:
            raise self._error(
                stage="store_chunks",
                message="文档向量写入持久化 Chroma 失败",
            ) from exc
        return chunks

    def close(self) -> None:
        """释放 Chroma、Embedding 模型引用和设备缓存，但保留磁盘数据。"""

        embedding_model = getattr(self, "embedding_model", None)
        if embedding_model is not None:
            try:
                embedding_model.to("cpu")
            except Exception:
                pass

        chroma_client = getattr(self, "chroma_client", None)
        if chroma_client is not None:
            try:
                chroma_client.close()
                chroma_client.clear_system_cache()
            except Exception:
                pass

        self.policy_chunk_collection = None
        self.chroma_client = None
        self.embedding_model = None
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _read_policies(path: str | Path) -> list[dict[str, str]]:
        """读取并做最小结构校验，返回政策 JSON 中的文档字段。"""

        try:
            file_path = Path(path)
            if not file_path.is_file():
                raise FileNotFoundError(file_path)
            payload: Any = json.loads(file_path.read_text(encoding="utf-8"))
        except JSONDecodeError as exc:
            raise IngestionService._error(
                stage="parse_policies",
                message=f"政策 JSON 格式错误：第 {exc.lineno} 行第 {exc.colno} 列",
            ) from exc
        except (OSError, UnicodeError) as exc:
            raise IngestionService._error(
                stage="read_policies",
                message="政策 JSON 文件读取失败",
            ) from exc

        if not isinstance(payload, list):
            raise IngestionService._error(
                stage="validate_policies",
                message="政策 JSON 顶层结构必须是数组",
            )

        documents: list[dict[str, str]] = []
        required_fields = ("id", "title", "type", "date", "source", "content")
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise IngestionService._error(
                    stage="validate_policies",
                    message=f"第 {index} 条政策必须是 JSON 对象",
                    details={"document_index": index},
                )
            if any(
                not isinstance(item.get(field), str) or not item[field].strip()
                for field in required_fields
            ):
                raise IngestionService._error(
                    stage="validate_policies",
                    message=f"第 {index} 条政策缺少有效字段",
                    details={"document_index": index},
                )
            documents.append({field: item[field] for field in required_fields})
        return documents

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
