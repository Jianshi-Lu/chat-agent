"""实现从输入文档到向量数据库的知识入库流水线。

处理链路为：Markdown 文件 → 读取正文 → 文本分块 → Embedding 向量化 →
写入 Chroma collection。

对外提供 ``IngestionService``，负责执行上述入库和相似度查询流程；
提供``TextChunk``，用于表示分块后的文本、分块 ID 及来源 metadata。
"""

from .pipeline import IngestionService, TextChunk

__all__ = ["IngestionService", "TextChunk"]
