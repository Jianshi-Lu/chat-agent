"""对话 Agent 的内部实现包。

本包提供模型调用适配器，供 ``app.service.client`` 组装默认对话服务使用。
上层业务不应绕过 service 入口直接依赖具体 Agent。
"""

from .agent import LangChainChatAgent

__all__ = ["LangChainChatAgent"]
