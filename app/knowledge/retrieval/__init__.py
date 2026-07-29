"""检索领域包。

本包负责定义不同检索方案之间共享的输入输出契约。当前提供统一的
``Retriever`` 调用协议和 ``RetrievedDocument`` 返回模型；具体的关键词、
BM25、向量和混合检索器会在后续模块中实现。
"""

from .base import Retriever
from .schemas import RetrievedDocument

__all__ = ["RetrievedDocument", "Retriever"]
