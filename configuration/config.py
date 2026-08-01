"""定义本地 OpenAI-compatible 对话服务配置。

配置直接写在本模块中，默认指向本机运行的 OpenAI-compatible 服务。本模块保存
HTTP、知识库、检索和模型客户端参数，不创建网络连接，也不保存对话历史。
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    system_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class HttpServerSettings:
    """承载本地 HTTP Server 的监听配置。"""

    host: str = "127.0.0.1"
    port: int = 8082


@dataclass(frozen=True, slots=True)
class KnowledgeRuntimeSettings:
    """承载知识库运行时所需的数据、模型和向量库路径。"""

    policies_path: Path = PROJECT_ROOT / "data" / "policies.json"
    embedding_model_path: Path = Path(
        "/Users/lujianshi/Desktop/models/embedding/Qwen:Qwen3-Embedding-8B"
    )
    chroma_path: Path = PROJECT_ROOT / "data" / "chroma"


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """承载知识检索策略参数。"""

    top_k: int = 3


@lru_cache(maxsize=1)
def get_chat_model_settings() -> ChatModelSettings:
    """返回进程内复用的对话模型配置实例。"""

    return ChatModelSettings()


@lru_cache(maxsize=1)
def get_http_server_settings() -> HttpServerSettings:
    """返回进程内复用的 HTTP Server 配置实例。"""

    return HttpServerSettings()


@lru_cache(maxsize=1)
def get_knowledge_runtime_settings() -> KnowledgeRuntimeSettings:
    """返回进程内复用的知识库运行时配置实例。"""

    return KnowledgeRuntimeSettings()


@lru_cache(maxsize=1)
def get_retrieval_settings() -> RetrievalSettings:
    """返回进程内复用的检索配置实例。"""

    return RetrievalSettings()


__all__ = [
    "ChatModelSettings",
    "HttpServerSettings",
    "KnowledgeRuntimeSettings",
    "RetrievalSettings",
    "get_chat_model_settings",
    "get_http_server_settings",
    "get_knowledge_runtime_settings",
    "get_retrieval_settings",
]
