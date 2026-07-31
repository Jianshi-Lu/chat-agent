"""Keyword 检索实现包。

当前包先提供政府采购领域的 ``KEYWORD_GROUPS`` 词表；后续由
``retriever.py`` 使用这些概念组完成匹配文本格式化、候选匹配、评分和排序。
本包不读取本地数据文件，也不负责过滤、去重或调用大模型，当前位于归档路径。
"""

from .retriever import KeywordRetriever

__all__ = ["KeywordRetriever"]
