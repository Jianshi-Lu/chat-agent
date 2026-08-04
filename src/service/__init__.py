"""提供政策知识入库和本地 Agent/RAG 对话相关的应用服务入口。

知识入库由 ``src.service.ingestion.IngestionPipeline`` 组装格式解析、切分、Embedding
和 Chroma 写入；独立任务入口是 ``python -m src.service.ingestion.main``。HTTP
Server 通过 ``python -m src.api.server`` 启动；FastAPI 生命周期创建 Runtime，复用
持久化 Chroma 和官方 Retriever，再由 ``agent`` 包通过 LangChain 编排模型和工具。
服务退出时释放进程资源，但保留磁盘上的向量数据。
"""

from .runtime import Runtime

__all__ = ["Runtime"]
