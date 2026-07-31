"""检索领域包。

本归档包保留不同检索方案共享的输入输出契约、``Retriever`` 调用协议、
``RetrievedDocument`` 返回模型和已实现的 ``KeywordRetriever``。BM25、向量
和混合检索器尚未实现，当前聊天入口不依赖本包。
"""

# 以下目录只记录恢复检索开发时可能继续扩展的结构。
# app/archive/retrieval/
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
