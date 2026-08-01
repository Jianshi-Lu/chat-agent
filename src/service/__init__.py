"""提供政策知识入库和本地 Agent/RAG 对话相关的应用服务入口。

上层可以从 ``src.service.ingestion`` 导入 ``IngestionService``。执行
``python -m src.service`` 会启动 HTTP Server；FastAPI 生命周期创建 Runtime，
在持久化 Chroma 中完成分块和向量入库，由平级的 ``retrieval`` 包完成问题检索，
再由 ``agent`` 包通过 LangChain 编排模型和工具。服务退出时释放进程资源，
但保留磁盘上的向量数据。
"""

from .runtime import Runtime

__all__ = ["Runtime"]
