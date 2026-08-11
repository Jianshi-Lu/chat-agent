"""提供本地 Agent/RAG 对话服务入口。

HTTP Server 通过 ``python -m src.api.server`` 启动；FastAPI 生命周期创建 Runtime，
复用对话模型、MySQL Checkpointer 和独立 vector-service 客户端。文档切分、Embedding、
Chroma 入库和检索均由 vector-service 负责。
"""

from .runtime import Runtime

__all__ = ["Runtime"]
