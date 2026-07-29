"""检索领域包。

本包负责定义不同检索方案共享的输入输出契约。当前提供统一的
``Retriever`` 调用协议和 ``RetrievedDocument`` 返回模型；具体的关键词、
BM25、向量和混合检索器会在后续模块中实现。
"""

# TODO：按以下结构逐步补充具体检索实现；当前包只提供公共契约出口。
# app/retrieval/
# ├── __init__.py                 → 公共出口
# ├── base.py                     → Retriever Protocol
# ├── schemas.py                  → RetrievedDocument
# ├── keyword/
# │   ├── __init__.py             → 导出 KeywordRetriever
# │   ├── terms.py                → 领域概念与同义词
# │   └── retriever.py            → Keyword 实现
# ├── vector/
# │   ├── __init__.py             → 导出 VectorRetriever
# │   ├── retriever.py            → Embedding 检索
# │   ├── chunker.py              → 长文档分块
# │   └── store.py                → 向量存储适配
# ├── hybrid/
# │   ├── __init__.py             → 导出 HybridRetriever
# │   └── retriever.py            → Keyword + Vector 排名融合
# ├── bm25/                       → 后续可选词法检索基线
# └── router.py                   → 根据策略选择检索器
from .base import Retriever
from .schemas import RetrievedDocument

__all__ = ["RetrievedDocument", "Retriever"]
