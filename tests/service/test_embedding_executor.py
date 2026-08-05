"""验证批量编码和 MPS 固定线程执行边界。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock, get_ident
from time import sleep
from typing import Any, cast

import numpy as np
import pytest

from src.service.ingestion.embedding import (
    BatchHuggingFaceEmbeddings,
    MpsEmbeddingExecutor,
)


class FakeSentenceTransformer:
    """记录官方 HuggingFace 包装器最终发出的 encode 调用。"""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append((list(texts), kwargs))
        return np.asarray(
            [[float(index), float(index + 1)] for index in range(len(texts))]
        )


def test_batch_huggingface_embeds_all_queries_with_one_encode_call() -> None:
    """N 个查询必须传给一次底层 SentenceTransformer.encode。"""

    client = FakeSentenceTransformer()
    embeddings = BatchHuggingFaceEmbeddings.model_construct(
        model_name="fake",
        cache_folder=None,
        model_kwargs={},
        encode_kwargs={"normalize_embeddings": False},
        query_encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 4,
            "prompt_name": "query",
        },
        multi_process=False,
        show_progress=False,
    )
    object.__setattr__(embeddings, "_client", client)

    vectors = embeddings.embed_queries(["问题一", "问题二", "问题三"])

    assert len(vectors) == 3
    assert len(client.calls) == 1
    assert client.calls[0][0] == ["问题一", "问题二", "问题三"]
    assert client.calls[0][1]["batch_size"] == 4
    assert client.calls[0][1]["prompt_name"] == "query"


class RecordingEmbedder:
    """记录模型操作的线程和同一时刻活跃调用数。"""

    def __init__(self) -> None:
        self.lock = Lock()
        self.active = 0
        self.max_active = 0
        self.call_thread_ids: set[int] = set()
        self.batch_calls: list[list[str]] = []
        self.close_thread_id: int | None = None

    def _record(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.call_thread_ids.add(get_ident())
        try:
            sleep(0.01)
        finally:
            with self.lock:
                self.active -= 1

    def embed_query(self, text: str) -> list[float]:
        self._record()
        return [float(len(text))]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._record()
        return [[float(len(text))] for text in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        self._record()
        return [[float(len(text))] for text in texts]

    def close(self) -> None:
        self.close_thread_id = get_ident()


def test_mps_executor_uses_one_thread_for_create_calls_and_close() -> None:
    """并发调用方只能向队列提交任务，不能并发进入模型。"""

    created: dict[str, object] = {}

    def factory() -> BatchHuggingFaceEmbeddings:
        delegate = RecordingEmbedder()
        created["thread_id"] = get_ident()
        created["delegate"] = delegate
        return cast(Any, delegate)

    embeddings = MpsEmbeddingExecutor(factory)
    with ThreadPoolExecutor(max_workers=6) as callers:
        results = list(callers.map(embeddings.embed_query, [f"问题{i}" for i in range(12)]))
    document_vectors = embeddings.embed_documents(["文档一", "文档二"])
    query_vectors = embeddings.embed_queries(["子问题一", "子问题二", "子问题三"])
    worker_thread_id = embeddings.worker_thread_id
    embeddings.close()
    embeddings.close()

    delegate = cast(RecordingEmbedder, created["delegate"])
    assert len(results) == 12
    assert len(document_vectors) == 2
    assert len(query_vectors) == 3
    assert worker_thread_id is not None
    assert created["thread_id"] == worker_thread_id
    assert delegate.call_thread_ids == {worker_thread_id}
    assert delegate.close_thread_id == worker_thread_id
    assert delegate.max_active == 1
    assert delegate.batch_calls == [["子问题一", "子问题二", "子问题三"]]

    with pytest.raises(RuntimeError, match="已经关闭"):
        embeddings.embed_query("关闭后的请求")
