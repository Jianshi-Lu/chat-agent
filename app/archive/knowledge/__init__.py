"""知识文档领域包。

本包接收本地 JSON 文档数据，负责模型定义、字段校验和加载，向上层返回
``list[PolicyDocument]``；检索能力位于同级 ``app.archive.retrieval`` 包，分块、
向量化以及大模型调用不属于本包职责。
"""
