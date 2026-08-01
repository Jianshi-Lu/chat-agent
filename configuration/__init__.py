"""应用配置包。

本包集中保存当前应用的本地模型连接和生成参数。业务模块通过这里导入配置，
不再把配置文件放在具体业务包内部。
"""

from .config import (
    ChatSettings,
    ServerSettings,
    get_chat_settings,
    get_server_settings,
)

__all__ = [
    "ChatSettings",
    "ServerSettings",
    "get_chat_settings",
    "get_server_settings",
]
