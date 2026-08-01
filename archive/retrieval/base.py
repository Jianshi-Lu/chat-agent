"""定义不同检索方案共享的调用契约。

上层先通过 ``archive.knowledge.loader.load_documents`` 获取完整的
``PolicyDocument`` 集合，再调用具体检索器的 ``build`` 构建内部索引，最后
调用 ``retrieve`` 返回统一的 ``RetrievedDocument`` 列表。本模块只定义接口，
不读取文件、不实现检索算法，也不负责 Benchmark、排序规则或大模型调用。
"""

from collections.abc import Sequence
from typing import Protocol

from archive.knowledge.schemas import PolicyDocument

from .schemas import RetrievedDocument


class Retriever(Protocol):
    """所有检索方案必须遵守的统一接口。

    具体实现可以是 Keyword、BM25、Vector 或 Hybrid 检索器。实现类不要求
    显式继承本协议，只要提供相同的 ``name``、``build`` 和 ``retrieve``
    成员，就可以被类型检查器视为一个 ``Retriever``。

    统一调用链为：
    ``PolicyDocument`` 集合 → ``build`` → 内部索引 → ``retrieve`` →
    ``list[RetrievedDocument]``。

    实现类需要自行保证：查询前已经完成 ``build``，空查询返回空列表，
    ``top_k`` 为正数，以及所有加载和检索失败都通过项目统一的
    ``AppException`` 向上抛出。
    """

    name: str

    def build(self, documents: Sequence[PolicyDocument]) -> None:
        """接收完整文档集合，构建或刷新当前检索算法的内部索引。"""

        ...

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> list[RetrievedDocument]:
        """根据查询返回按当前算法排序的最多 ``top_k`` 条文档结果。"""

        ...
