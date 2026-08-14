"""定义对话 Agent、Checkpointer 和独立向量服务的连接配置。"""

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path


def _load_local_environment() -> None:
    """读取与本配置模块同目录的 .env，不覆盖进程或 IDE 已设置的变量。"""

    env_file = Path(__file__).with_name(".env")
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if name:
            os.environ.setdefault(name, value)


_load_local_environment()


@dataclass(frozen=True, slots=True)
class ChatModelSettings:
    """承载本地对话模型的连接和生成参数。

    默认值对应本机端口 `8000` 上的 OpenAI-compatible 服务及用户提供的模型。
    ``max_tokens``、``temperature`` 和 ``top_p`` 通过 ``ChatOpenAI`` 传给服务，
    ``top_k`` 与 ``repetition_penalty`` 通过 ``extra_body`` 传递给兼容服务。
    """

    base_url: str = "http://127.0.0.1:8000/v1"
    # 默认值仅用于未启用认证的本地兼容服务；实际 oMLX 密钥从环境变量注入。
    api_key: str = field(default="local", repr=False)
    model_name: str = "Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP"

    max_tokens: int = 131_072
    temperature: float = 0.5
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0

    timeout: float = 600.0
    max_retries: int = 0
    system_prompt: str | None = (
        "你是招投标平台资料分析助手。你可以在任意会话中查询公共知识文件、全部"
        "投标记录及其资信材料，不要求用户在创建会话时预先绑定投标记录。"
        "用户询问有哪些已入库文件时调用 list_files。用户提到具体文件名、"
        "近似文件名、带扩展名的文档，或结合上下文询问‘这个文件’时，必须先调用"
        " find_file_by_name，不能在查询前声称无法访问文件。若只匹配一个文件，"
        "知识文件必须继续调用 read_knowledge_file，资信文件必须继续调用"
        " read_credential_file，再根据正文回答；若匹配多个文件，列出文件类型、"
        "文件名和投标记录 ID，请用户确认，不能自行猜测。"
        "PUBLIC-E2E-8047 这类包含字母和连字符的知识编号属于公共知识检索词，"
        "绝不能把末尾数字当成投标记录 ID；只有用户明确写出‘投标记录 13’时，"
        "才可以调用 get_bid_record 或 retrieve_credential_files_by_bid_record_id。"
        "不指定文件名的公共知识问题调用 retrieve_knowledge_files。涉及投标记录、投标人或跨文件"
        "资信分析时使用投标资料工具：浏览全部记录调用 list_bid_records；明确投标"
        "记录 ID 时调用 get_bid_record；跨记录检索投标人、报价、项目、公告或风险"
        "事实时调用 retrieve_bid_records；跨记录检索资信正文时调用"
        " retrieve_credential_files；明确记录 ID 并检索其资信正文时调用"
        " retrieve_credential_files_by_bid_record_id。需要综合分析时可以依次调用多个不同工具，"
        "但不要用公共知识结果替代投标资料，也不要把私有投标资料当作公共知识。"
        "涉及资信材料的回答应明确说明所依据的投标记录 ID 和文件名，便于用户在"
        "界面中继续下载已检索到的资料。不要输出对象存储路径、内部 UUID、内部文件 ID 或自称可"
        "下载链接；文件下载由界面中的资信材料卡片完成。"
        "如果用户一轮提出多个问题，先拆成独立子问题，全部放入同一个 queries 数组，"
        "同一类检索只调用一次，不要为每个子问题分别调用检索工具；"
        "获得足够结果后直接回答，不要使用相同参数重复调用同一工具。"
        "只有在相应工具明确返回未找到或正文不足后，才说明资料不足；不要凭空补充"
        "事实，也不要虚构工具能力限制。回答尽量简洁，并指出文件名或投标记录来源。"
    )


@dataclass(frozen=True, slots=True)
class ContextCompressionSettings:
    """承载 LangChain 对话上下文压缩中间件参数。

    ``trigger_tokens`` 当前使用便于本地验证的默认值，实际部署时应结合模型上下文窗口、
    Tool schema 和输出预留空间调整。摘要后仅保留最近 8 条消息，为 4,000-token 触发线
    留出回落空间，避免每轮对已经摘要的会话重复摘要。``summary_model_name`` 为空时直接
    复用当前对话模型客户端；配置了独立模型名时，才创建独立摘要客户端并使用
    ``summary_temperature`` / ``summary_max_tokens``。
    """

    enabled: bool = True
    summary_model_name: str | None = None
    trigger_tokens: int = 4_000
    keep_messages: int = 8
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
class CheckpointSettings:
    """承载 LangGraph Checkpointer 的 MySQL 连接配置。

    ``database_url`` 使用 ``mysql://user:password@host:port/database`` 格式，
    默认指向本地开发用的虚拟机独立库；也可用环境变量
    ``CHAT_AGENT_CHECKPOINT_DATABASE_URL`` 覆盖。该库必须与业务库 ``chat_agent``
    分离，避免 Checkpointer 表与业务表混用。Checkpointer 持久化始终开启，
    不提供关闭开关。
    """

    database_url: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class HttpServerSettings:
    """承载本地 HTTP Server 的监听配置。"""

    host: str = "127.0.0.1"
    port: int = 8082
    log_level: str = "INFO"


@dataclass(frozen=True, slots=True)
class VectorServiceSettings:
    """承载 chat-agent 到独立 vector-service 的 HTTP 连接参数。"""

    base_url: str = "http://127.0.0.1:8083"
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("vector-service base_url 不能为空")
        if self.timeout <= 0:
            raise ValueError("vector-service timeout 必须大于 0")


_DEFAULT_CHAT_MODEL_SETTINGS = ChatModelSettings()


@lru_cache(maxsize=1)
def get_chat_model_settings() -> ChatModelSettings:
    """返回进程内复用的对话模型配置实例。"""

    return ChatModelSettings(
        base_url=(
            os.environ.get("OMLX_BASE_URL")
            or _DEFAULT_CHAT_MODEL_SETTINGS.base_url
        ),
        api_key=(
            os.environ.get("OMLX_API_KEY")
            or _DEFAULT_CHAT_MODEL_SETTINGS.api_key
        ),
    )


@lru_cache(maxsize=1)
def get_context_compression_settings() -> ContextCompressionSettings:
    """返回进程内复用的上下文压缩配置实例。"""

    return ContextCompressionSettings()


@lru_cache(maxsize=1)
def get_checkpoint_settings() -> CheckpointSettings:
    """返回进程内复用的 Checkpointer 配置实例。"""

    return CheckpointSettings(
        database_url=os.environ.get("CHAT_AGENT_CHECKPOINT_DATABASE_URL") or None,
    )


@lru_cache(maxsize=1)
def get_http_server_settings() -> HttpServerSettings:
    """返回进程内复用的 HTTP Server 配置实例。"""

    return HttpServerSettings(
        log_level=os.environ.get("CHAT_AGENT_LOG_LEVEL", "INFO").upper(),
    )


@lru_cache(maxsize=1)
def get_vector_service_settings() -> VectorServiceSettings:
    """返回独立向量服务连接配置。"""

    timeout = os.environ.get("VECTOR_SERVICE_TIMEOUT")
    return VectorServiceSettings(
        base_url=(
            os.environ.get("VECTOR_SERVICE_BASE_URL")
            or "http://127.0.0.1:8083"
        ).rstrip("/"),
        timeout=float(timeout) if timeout and timeout.strip() else 120.0,
    )


__all__ = [
    "ChatModelSettings",
    "CheckpointSettings",
    "ContextCompressionSettings",
    "HttpServerSettings",
    "VectorServiceSettings",
    "get_chat_model_settings",
    "get_checkpoint_settings",
    "get_context_compression_settings",
    "get_http_server_settings",
    "get_vector_service_settings",
]
