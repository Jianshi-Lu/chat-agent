"""创建线程安全的本地 LangChain Embeddings 运行时。

``create_model`` 是本项目对外暴露的动态路由工厂：普通设备直接返回
``BatchHuggingFaceEmbeddings``，MPS 则返回 ``MpsQueueEmbeddings``。
后者实现与普通 Embeddings 相同的调用接口，只额外增加 MPS 单线程排队能力。
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from configuration import get_ingestion_settings
from src.errors import AppException, KNOWLEDGE_INGESTION_FAILED
from src.service.agent.mps_queue import MpsQueueEmbeddings


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


# 类型上，两个类都是 Embeddings 的具体实现，也是 create_model 的并列返回类型；
# 运行时职责上，MpsQueueEmbeddings 会组合并包装 BatchHuggingFaceEmbeddings，
# 只为 Apple MPS 补充官方封装没有提供的固定线程串行化和生命周期线程亲和。
# 因此调用方不需要根据设备区分两种实现。
EmbeddingRuntime = BatchHuggingFaceEmbeddings | MpsQueueEmbeddings


def create_model(
    model_path: str | Path,
    *,
    device: str | None = None,
    embedding_batch_size: int | None = None,
    query_embedding_batch_size: int | None = None,
) -> EmbeddingRuntime:
    """校验模型并按设备动态返回具体 Embeddings 实现。

    ``create_model`` 名称表示统一的模型运行时创建入口；它的返回值不一定是
    裸的 ``BatchHuggingFaceEmbeddings``，MPS 环境下会是实现同样接口的
    ``MpsQueueEmbeddings``。
    """

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

    def _build_model() -> BatchHuggingFaceEmbeddings:
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
            # 入口：传入工厂函数而不是提前执行，确保底层模型在 MPS 工作线程创建。
            return MpsQueueEmbeddings(_build_model)
        return _build_model()
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
        operation="create_model",
        stage=stage,
        message=message,
        details=details,
    )


__all__ = [
    "BatchHuggingFaceEmbeddings",
    "EmbeddingRuntime",
    "create_model",
]
