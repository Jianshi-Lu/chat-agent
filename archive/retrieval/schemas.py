"""定义检索器之间共享的结果模型。

检索器接收用户查询和已经加载的 ``PolicyDocument``，完成候选文档排序后，
通过 ``RetrievedDocument`` 向上层返回统一结构。该模块只负责结果结构和
类型校验，不负责关键词匹配、索引构建、向量计算或大模型调用。当前模型随
检索实现一并归档，不属于聊天入口的运行依赖。
"""

from datetime import date

from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    """检索器统一返回的一条文档结果。

    所有检索算法都必须返回相同的业务字段，保证上层上下文组装和后续 API
    不需要感知具体算法。``score`` 只用于同一种算法内部对相关度排序，不
    表示不同算法之间可以直接比较的绝对相关性。

    输入来自具体检索器从 ``PolicyDocument`` 提取的元数据、Markdown 正文和
    当前算法计算出的排序分数；输出是字段类型稳定、日期为
    ``datetime.date`` 的结果模型，正文保持原始 Markdown。
    """

    document_id: str
    title: str
    type: str
    date: date
    source: str
    content: str
    score: float
