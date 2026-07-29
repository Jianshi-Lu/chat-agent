"""定义检索器之间共享的结果模型。

检索器接收用户查询和已经加载的 ``PolicyDocument``，完成候选文档排序后，
通过 ``RetrievedDocument`` 向上层返回统一结构。该模块只负责结果结构和
类型校验，不负责关键词匹配、索引构建、向量计算或大模型调用。
"""

from datetime import date

from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    """检索器统一返回的一条文档结果。

    所有检索算法都必须返回相同的业务字段，保证上层 Benchmark、上下文组装
    和后续 API 不需要感知具体算法。

    ``score`` 作用于同一种算法内部对相关度的排序，以截取 Top-K，
    不表示不同算法之间可以直接比较的绝对相关性。

    输入：
        检索器从 ``PolicyDocument`` 提取文档元数据、Markdown 正文和本算法
        计算出的排序分数。

    输出：
        一个经过 Pydantic 类型校验的结果对象，其中 ``date`` 为
        ``datetime.date``，``content`` 保留原始 Markdown 内容。

    处理链路：
        ``PolicyDocument`` → 具体检索器排序 → ``RetrievedDocument`` →
        Benchmark 或上层业务调用方。
    """

    document_id: str
    title: str
    type: str
    date: date
    source: str
    content: str
    score: float
