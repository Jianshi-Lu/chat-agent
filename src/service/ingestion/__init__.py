"""实现从政策 JSON 到向量数据库的知识入库流水线。

处理链路为：``data/policies.json`` → 读取政策正文 → 文本分块 → Embedding
向量化 → 写入当前进程内的 Chroma collection。服务退出后 collection 和
模型引用都会释放，不产生持久化向量数据库文件。

对外提供 ``IngestionService``，负责执行上述入库流程；相似度查询由平级的
``service.retrieval`` 包负责。另提供 ``TextChunk``，用于表示分块后的文本、
分块 ID 及来源 metadata。
"""

from .pipeline import (
    DEFAULT_POLICIES_PATH,
    IngestionService,
    TextChunk,
)

__all__ = ["DEFAULT_POLICIES_PATH", "IngestionService", "TextChunk"]
