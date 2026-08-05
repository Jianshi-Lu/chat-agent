"""验证批量查询编码、逐向量搜索和 Agent Tool 边界。"""

import pytest
from langchain_core.documents import Document

from src.errors import AppException
from src.service.agent.tools import build_retriever_tool
from src.service.retrieval import BatchRetriever


class RecordingBatchEmbeddings:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [
            [float(index), float(index + 1)]
            for index in range(len(texts))
        ]


class RecordingVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[float], int]] = []

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        *,
        k: int,
    ) -> list[Document]:
        self.calls.append((embedding, k))
        index = int(embedding[0]) + 1
        return [
            Document(
                id=f"policy-{index}",
                page_content=f"第 {index} 个查询的政策正文",
                metadata={"id": f"policy-{index}", "source": "财政部"},
            )
        ]


def _build_batch_retriever() -> tuple[
    BatchRetriever,
    RecordingBatchEmbeddings,
    RecordingVectorStore,
]:
    embeddings = RecordingBatchEmbeddings()
    vector_store = RecordingVectorStore()
    retriever = BatchRetriever(
        vector_store=vector_store,  # type: ignore[arg-type]
        embeddings=embeddings,
        top_k=3,
    )
    return retriever, embeddings, vector_store


def test_retrieve_many_embeds_once_then_searches_each_vector() -> None:
    """三个查询应形成一次编码调用和三次独立向量搜索。"""

    retriever, embeddings, vector_store = _build_batch_retriever()

    results = retriever.retrieve_many(["问题一", "问题二", "问题三"])

    assert embeddings.calls == [["问题一", "问题二", "问题三"]]
    assert vector_store.calls == [
        ([0.0, 1.0], 3),
        ([1.0, 2.0], 3),
        ([2.0, 3.0], 3),
    ]
    assert [result.query for result in results] == ["问题一", "问题二", "问题三"]
    assert [result.documents[0].id for result in results] == [
        "policy-1",
        "policy-2",
        "policy-3",
    ]


def test_build_retriever_tool_accepts_one_queries_array_and_formats_groups() -> None:
    retriever, embeddings, vector_store = _build_batch_retriever()
    tool = build_retriever_tool(retriever)

    result = tool.invoke({"queries": ["公告是什么？", "发布时间是什么？"]})

    assert tool.name == "retrieve_many"
    assert embeddings.calls == [["公告是什么？", "发布时间是什么？"]]
    assert len(vector_store.calls) == 2
    assert "查询 1：公告是什么？" in result
    assert "查询 2：发布时间是什么？" in result
    assert "文档ID：policy-1" in result
    assert "来源：财政部" in result
    assert "正文：" in result


def test_retrieve_many_rejects_blank_query_before_embedding() -> None:
    retriever, embeddings, vector_store = _build_batch_retriever()

    with pytest.raises(AppException) as exc_info:
        retriever.retrieve_many(["有效问题", " "])

    assert exc_info.value.stage == "validate_queries"
    assert embeddings.calls == []
    assert vector_store.calls == []
