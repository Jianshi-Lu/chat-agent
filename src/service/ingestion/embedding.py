"""创建本地 Embedding 模型对应的官方 LangChain Embeddings 实例。"""

from pathlib import Path

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from configuration import get_ingestion_settings
from src.errors import AppException, KNOWLEDGE_INGESTION_FAILED


def create_embedders(
    model_path: str | Path,
    *,
    device: str | None = None,
    embedding_batch_size: int | None = None,
    query_embedding_batch_size: int | None = None,
) -> HuggingFaceEmbeddings:
    """校验本地模型并创建 ``HuggingFaceEmbeddings``。

    batch size 未显式传入时，使用 ``configuration.config`` 中的入库配置。
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

    try:
        return HuggingFaceEmbeddings(
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


__all__ = ["create_embedders"]
