"""定义本地 OpenAI-compatible 对话服务配置。

配置直接写在本模块中，默认指向本机运行的 OpenAI-compatible 服务。本模块保存
HTTP、知识库、检索、模型客户端和上下文压缩参数，不创建网络连接。
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THREAD_ID = "default"


@dataclass(frozen=True, slots=True)
class ChatModelSettings:
    """承载本地对话模型的连接和生成参数。

    默认值对应本机端口 `8000` 上的 OpenAI-compatible 服务及用户提供的模型。
    ``context_window`` 是服务端上下文窗口记录，不会作为普通采样参数发送；
    ``max_tokens``、``temperature`` 和 ``top_p`` 通过 ``ChatOpenAI`` 传给服务，
    ``top_k`` 与 ``repetition_penalty`` 通过 ``extra_body`` 传递给兼容服务。
    """

    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "20020228"
    model_name: str = "Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP"

    context_window: int = 262_144
    max_tokens: int = 131_072
    temperature: float = 0.5
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0

    timeout: float = 600.0
    max_retries: int = 0
    system_prompt: str | None = (
        "你是政策资料问答助手。涉及政策事实、政策定义、政策要求或政策来源的问题，"
        "必须优先调用 retrieve_many 获取知识库片段，再依据检索结果回答。"
        "如果用户一轮提出多个问题，先拆成独立子问题，全部放入同一个 queries 数组，"
        "只调用一次 retrieve_many，不要为每个子问题分别调用检索工具；"
        "如果上下文不足以支持结论，必须明确说明知识库中没有足够信息，"
        "不要凭空补充事实。已注册的工具可能包含占位实现，不能把占位结果"
        "当作真实政策事实。回答尽量简洁，并在适合时指出资料来源。"
    )


@dataclass(frozen=True, slots=True)
class ContextCompressionSettings:
    """承载 LangChain 对话上下文压缩中间件参数。

    ``trigger_tokens`` 当前使用便于本地验证的默认值，实际部署时应结合模型上下文窗口、
    Tool schema 和输出预留空间调整。``summary_model_name`` 为空时复用主模型名称，
    但仍会以独立的摘要客户端调用。
    """

    enabled: bool = True
    summary_model_name: str | None = None
    trigger_tokens: int = 4_000
    keep_messages: int = 20
    summary_max_tokens: int = 4_096
    summary_temperature: float = 0.0

    def __post_init__(self) -> None:
        """在 Agent 创建前拒绝无效的压缩参数。"""

        if self.trigger_tokens <= 0:
            raise ValueError("trigger_tokens 必须大于 0")
        if self.keep_messages <= 0:
            raise ValueError("keep_messages 必须大于 0")
        if self.summary_max_tokens <= 0:
            raise ValueError("summary_max_tokens 必须大于 0")


@dataclass(frozen=True, slots=True)
class HttpServerSettings:
    """承载本地 HTTP Server 的监听配置。"""

    host: str = "127.0.0.1"
    port: int = 8082


@dataclass(frozen=True, slots=True)
class IngestionSettings:
    """承载知识入库所需的数据、模型、向量库和处理参数。"""

    source_path: Path = PROJECT_ROOT / "data" / "policies.json"
    embedding_model_path: Path = Path(
        "/Users/lujianshi/Desktop/models/embedding/Qwen:Qwen3-Embedding-8B"
    )
    chroma_path: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "knowledge_file_chunks"
    chunk_size: int = 300
    chunk_overlap: int = 50
    embedding_batch_size: int = 4
    query_embedding_batch_size: int = 4


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """承载知识检索策略参数。"""

    top_k: int = 3


@lru_cache(maxsize=1)
def get_chat_model_settings() -> ChatModelSettings:
    """返回进程内复用的对话模型配置实例。"""

    return ChatModelSettings()


@lru_cache(maxsize=1)
def get_context_compression_settings() -> ContextCompressionSettings:
    """返回进程内复用的上下文压缩配置实例。"""

    return ContextCompressionSettings()


@lru_cache(maxsize=1)
def get_http_server_settings() -> HttpServerSettings:
    """返回进程内复用的 HTTP Server 配置实例。"""

    return HttpServerSettings()


@lru_cache(maxsize=1)
def get_ingestion_settings() -> IngestionSettings:
    """返回进程内复用的知识入库配置实例。"""

    return IngestionSettings()


@lru_cache(maxsize=1)
def get_retrieval_settings() -> RetrievalSettings:
    """返回进程内复用的检索配置实例。"""

    return RetrievalSettings()


__all__ = [
    "ChatModelSettings",
    "ContextCompressionSettings",
    "HttpServerSettings",
    "IngestionSettings",
    "RetrievalSettings",
    "DEFAULT_THREAD_ID",
    "get_chat_model_settings",
    "get_context_compression_settings",
    "get_http_server_settings",
    "get_ingestion_settings",
    "get_retrieval_settings",
]
