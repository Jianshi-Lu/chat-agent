"""政策知识入库模块。

``IngestionPipeline`` 只负责组装解析、切分、Embedding 和 Chroma；命令行任务
入口位于 ``service.ingestion.main``。
"""

from .pipeline import IngestionPipeline
from .models import KnowledgeDocument

__all__ = ["IngestionPipeline", "KnowledgeDocument"]
