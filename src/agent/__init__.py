"""对话 Agent 的内部实现包。

本包提供模型调用适配器，供归档的 ``archive.client`` 使用。
当前主流程由 ``src.service.runtime`` 编排 RAG 对话，上层业务不应绕过
service 入口直接依赖具体 Agent。
"""

from .agent import LangChainChatAgent

__all__ = ["LangChainChatAgent"]
