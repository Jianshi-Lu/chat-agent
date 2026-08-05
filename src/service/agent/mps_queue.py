"""提供 MPS Embeddings 的单线程消息队列和生命周期管理。

类型关系上，``MpsQueueEmbeddings`` 与 ``BatchHuggingFaceEmbeddings`` 都是
LangChain ``Embeddings`` 的具体实现，也是 ``create_model`` 可能返回的两个
并列运行时类型；职责关系上二者并不重复：``BatchHuggingFaceEmbeddings``
负责真正加载 Hugging Face 模型并生成向量，``MpsQueueEmbeddings`` 则通过组合
包装一个这样的 Embeddings delegate，只增加 MPS 单线程排队和线程亲和能力。

当前项目运行在 Apple MPS 设备上，共享 SentenceTransformer 模型曾被多个调用
线程同时进入并触发 PyTorch MPS 原生层崩溃。官方 ``HuggingFaceEmbeddings``
负责模型加载和 ``encode`` 调用，但不负责把外部并发请求串行到同一个固定线程，
也不保证模型创建、推理和释放发生在同一线程。因此本模块没有重新实现官方的
向量化逻辑，而是复用官方实现作为 delegate，并补充当前设备需要的并发治理层。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, get_ident
from typing import Any, cast

from langchain_core.embeddings import Embeddings


class MpsQueueEmbeddings(Embeddings):
    """与普通 Embeddings 并列使用的自定义 MPS 运行时实现。

    本类直接实现 LangChain ``Embeddings`` 接口，因此调用方可以像使用
    ``BatchHuggingFaceEmbeddings`` 一样调用它；但它自身不计算向量，而是
    持有由 ``factory`` 创建的具体 Embeddings delegate，并把全部模型操作
    转发到唯一固定工作线程。

    选择自定义包装器而不是直接使用官方封装，是因为当前 Apple MPS 环境需要
    额外满足两个官方封装未提供的运行时约束：并发调用必须串行执行，且模型的
    创建、推理和关闭必须保持在同一工作线程。CUDA 和 CPU 不需要这层 MPS
    保护时，``create_model`` 会直接返回 ``BatchHuggingFaceEmbeddings``。
    """

    def __init__(
        self,
        factory: Callable[[], Embeddings],
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mps-embedding",
        )
        self._lifecycle_lock = Lock()
        self._closed = False
        self._delegate: Embeddings | None = None
        self._worker_thread_id: int | None = None

        try:
            self._delegate, self._worker_thread_id = self._executor.submit(
                self._create_delegate,
                factory,
            ).result()
        except BaseException:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise

        print(
            f"MPS Embedding 固定执行线程已就绪：{self._worker_thread_id}",
            flush=True,
        )

    @staticmethod
    def _create_delegate(
        factory: Callable[[], Embeddings],
    ) -> tuple[Embeddings, int]:
        """在最终承接推理的同一线程中创建模型实例。"""

        return factory(), get_ident()

    @property
    def worker_thread_id(self) -> int | None:
        """返回固定 MPS 工作线程 ID，供日志与回归测试确认线程亲和性。"""

        return self._worker_thread_id

    def _invoke(self, method_name: str, *args: object) -> Any:
        delegate = self._delegate
        if delegate is None:
            raise RuntimeError("MPS Embedding 队列已经关闭")
        return getattr(delegate, method_name)(*args)

    def _submit(self, method_name: str, *args: object) -> Any:
        """原子检查生命周期并把一次模型操作加入单工作线程队列。"""

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MPS Embedding 队列已经关闭")
            future = self._executor.submit(self._invoke, method_name, *args)
        return future.result()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """串行执行文档批量编码。"""

        return cast(list[list[float]], self._submit("embed_documents", texts))

    def embed_query(self, text: str) -> list[float]:
        """串行执行单查询编码。"""

        return cast(list[float], self._submit("embed_query", text))

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """串行执行一次多查询批量编码。"""

        return cast(list[list[float]], self._submit("embed_queries", texts))

    def _close_delegate(self) -> None:
        delegate = self._delegate
        if delegate is not None:
            close = getattr(delegate, "close", None)
            if callable(close):
                close()
        self._delegate = None

    def close(self) -> None:
        """等待已排队任务完成，再在固定线程释放模型并停止工作线程。"""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            future = self._executor.submit(self._close_delegate)

        try:
            future.result()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)


__all__ = ["MpsQueueEmbeddings"]
