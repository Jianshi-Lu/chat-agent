"""创建 Markdown 使用的官方 LangChain 定长文本切分器。"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from configuration import get_ingestion_settings
from src.errors import AppException, KNOWLEDGE_INGESTION_FAILED


def create_splitter(
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """创建按字符长度切分 Markdown 的官方切分器。

    未显式传入参数时，使用 ``configuration.config`` 中的知识库运行时配置；
    显式参数仅用于任务级覆盖。
    """

    settings = get_ingestion_settings()
    chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap

    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise AppException(
            KNOWLEDGE_INGESTION_FAILED,
            module="service.ingestion.splitter",
            operation="create_splitter",
            stage="validate_chunks",
            message="chunk_size 必须大于 0，chunk_overlap 必须小于 chunk_size",
            details={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[""],
        keep_separator=False,
    )


__all__ = ["create_splitter"]
