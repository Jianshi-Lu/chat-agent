"""提供服务主链路使用的无状态 LangChain Agent。

本包负责把对话模型和工具定义组合成 Agent。工具当前只打印调用信息并返回
占位结果，后续再接入精确段落检索、语义检索等真实实现。
"""

from .agent import LangChainChatAgent

__all__ = ["LangChainChatAgent"]
