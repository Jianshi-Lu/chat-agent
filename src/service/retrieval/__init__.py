"""提供与知识入库流水线解耦的向量检索能力。

本包接收已经由 ``service.ingestion`` 打开并写入的 Chroma collection，
负责把用户问题向量化并返回相关文本片段、metadata 和距离。它不读取政策
文件、不负责创建数据库，也不负责把检索结果交给大模型。
"""

from .search import VectorRetriever

__all__ = ["VectorRetriever"]
