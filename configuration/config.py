"""定义本地 OpenAI-compatible 对话服务配置。

配置直接写在本模块中，默认指向本机运行的 OpenAI-compatible 服务。本模块只
保存模型客户端参数，不创建网络连接，也不保存对话历史。
"""

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class ChatSettings:
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


@lru_cache(maxsize=1)
def get_chat_settings() -> ChatSettings:
    """返回进程内复用的对话服务配置实例。"""

    return ChatSettings()


__all__ = ["ChatSettings", "get_chat_settings"]
