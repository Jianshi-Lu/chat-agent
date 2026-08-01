"""提供政策知识入库和本地 RAG 对话相关的应用服务入口。

上层可以从 ``src.service.ingestion`` 导入 ``IngestionService``。执行
``python -m src.service`` 会启动 HTTP Server；FastAPI 生命周期创建 Runtime，
在内存 Chroma 中完成分块和向量入库，由平级的 ``retrieval`` 包完成问题检索，
再通过 LangChain 调用本地模型。服务不持久化向量库，退出时释放内存。
"""

from .runtime import Runtime

__all__ = ["Runtime"]
