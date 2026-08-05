"""创建线程安全的本地 LangChain Embeddings 运行时。"""

from __future__ import annotations

import gc
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock, get_ident
from typing import Any, cast

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from configuration import get_ingestion_settings
from src.errors import AppException, KNOWLEDGE_INGESTION_FAILED


class BatchHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """在官方实现上补充一次编码多个查询和显式释放能力。"""

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """使用查询 prompt 在一次 ``encode`` 调用中编码全部查询。"""

        if not texts:
            return []
        encode_kwargs = (
            self.query_encode_kwargs
            if self.query_encode_kwargs
            else self.encode_kwargs
        )
        return self._embed(texts, encode_kwargs)

    def close(self) -> None:
        """释放模型及对应设备缓存；MPS 包装器会在固定线程调用本方法。"""

        client = getattr(self, "_client", None)
        if client is not None:
            try:
                if str(getattr(client, "device", "")).startswith("mps"):
                    torch.mps.synchronize()
            except Exception:
                pass
            try:
                client.to("cpu")
            except Exception:
                pass

        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class MpsEmbeddingExecutor(Embeddings):
    """把共享 MPS 模型的全部操作排队到唯一固定工作线程。"""

    def __init__(
        self,
        factory: Callable[[], BatchHuggingFaceEmbeddings],
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mps-embedding",
        )
        self._lifecycle_lock = Lock()
        self._closed = False
        self._delegate: BatchHuggingFaceEmbeddings | None = None
        self._worker_thread_id: int | None = None

        try:
            self._delegate, self._worker_thread_id = self._executor.submit(
                self._create_delegate,
                factory,
            ).result()
        except BaseException:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise

        print(
            f"MPS Embedding 固定执行线程已就绪：{self._worker_thread_id}",
            flush=True,
        )

    @staticmethod
    def _create_delegate(
        factory: Callable[[], BatchHuggingFaceEmbeddings],
    ) -> tuple[BatchHuggingFaceEmbeddings, int]:
        """在最终承接推理的同一线程中创建模型实例。"""

        return factory(), get_ident()

    @property
    def worker_thread_id(self) -> int | None:
        """返回固定 MPS 工作线程 ID，供日志与回归测试确认线程亲和性。"""

        return self._worker_thread_id

    def _invoke(self, method_name: str, *args: object) -> Any:
        delegate = self._delegate
        if delegate is None:
            raise RuntimeError("MPS Embedding 执行器已经关闭")
        return getattr(delegate, method_name)(*args)

    def _submit(self, method_name: str, *args: object) -> Any:
        """原子检查生命周期并把一次模型操作加入单工作线程队列。"""

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MPS Embedding 执行器已经关闭")
            future = self._executor.submit(self._invoke, method_name, *args)
        return future.result()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """串行执行文档批量编码。"""

        return cast(list[list[float]], self._submit("embed_documents", texts))

    def embed_query(self, text: str) -> list[float]:
        """串行执行单查询编码。"""

        return cast(list[float], self._submit("embed_query", text))

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """串行执行一次多查询批量编码。"""

        return cast(list[list[float]], self._submit("embed_queries", texts))

    def _close_delegate(self) -> None:
        delegate = self._delegate
        if delegate is not None:
            delegate.close()
        self._delegate = None

    def close(self) -> None:
        """等待已排队任务完成，再在固定线程释放模型并停止工作线程。"""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            future = self._executor.submit(self._close_delegate)

        try:
            future.result()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)


EmbeddingRuntime = BatchHuggingFaceEmbeddings | MpsEmbeddingExecutor


def create_embedders(
    model_path: str | Path,
    *,
    device: str | None = None,
    embedding_batch_size: int | None = None,
    query_embedding_batch_size: int | None = None,
) -> EmbeddingRuntime:
    """校验本地模型，并按设备创建直接或固定线程 Embeddings。"""

    settings = get_ingestion_settings()
    embedding_batch_size = (
        settings.embedding_batch_size
        if embedding_batch_size is None
        else embedding_batch_size
    )
    query_embedding_batch_size = (
        settings.query_embedding_batch_size
        if query_embedding_batch_size is None
        else query_embedding_batch_size
    )
    if embedding_batch_size <= 0 or query_embedding_batch_size <= 0:
        raise _error(
            stage="validate_config",
            message="Embedding batch size 必须大于 0",
            details={
                "embedding_batch_size": embedding_batch_size,
                "query_embedding_batch_size": query_embedding_batch_size,
            },
        )

    model_path = Path(model_path)
    if not model_path.is_dir():
        raise _error(
            stage="initialize_model",
            message="本地 Embedding 模型目录不存在",
        )

    missing_data_files = [
        file_path.name
        for file_path in model_path.glob("*.safetensors")
        if file_path.stat().st_size == 0 or file_path.stat().st_blocks == 0
    ]
    if missing_data_files:
        raise _error(
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

    def create_model() -> BatchHuggingFaceEmbeddings:
        return BatchHuggingFaceEmbeddings(
            model=str(model_path),
            model_kwargs={
                "device": device,
                "local_files_only": True,
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": embedding_batch_size,
            },
            query_encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": query_embedding_batch_size,
                "prompt_name": "query",
            },
            show_progress=False,
        )

    try:
        if device == "mps":
            return MpsEmbeddingExecutor(create_model)
        return create_model()
    except Exception as exc:
        raise _error(
            stage="initialize_model",
            message="本地 Embedding 模型加载失败",
        ) from exc


def _error(
    *,
    stage: str,
    message: str,
    details: dict[str, object] | None = None,
) -> AppException:
    """构造 Embedding 模块的结构化异常。"""

    return AppException(
        KNOWLEDGE_INGESTION_FAILED,
        module="service.ingestion.embedding",
        operation="create_embedders",
        stage=stage,
        message=message,
        details=details,
    )


__all__ = [
    "BatchHuggingFaceEmbeddings",
    "EmbeddingRuntime",
    "MpsEmbeddingExecutor",
    "create_embedders",
]
