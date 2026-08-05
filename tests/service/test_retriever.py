"""验证向量检索 Tool 的查询编码、逐向量搜索和 Agent 边界。"""

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from src.errors import AppException
from src.service.agent.tools.retrieve import RetrievalToolContext, retrieve


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


class ToolCallingFakeChatModel(FakeMessagesListChatModel):
    """让 Fake Chat Model 支持 Agent 的工具绑定。"""

    def bind_tools(self, tools, **kwargs):
        return self


def _build_context() -> tuple[
    RetrievalToolContext,
    RecordingBatchEmbeddings,
    RecordingVectorStore,
]:
    embeddings = RecordingBatchEmbeddings()
    vector_store = RecordingVectorStore()
    context = RetrievalToolContext(
        vector_store=vector_store,  # type: ignore[arg-type]
        embeddings=embeddings,
        top_k=3,
    )
    return context, embeddings, vector_store


def _build_tool_runtime(context: RetrievalToolContext) -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=lambda _: None,
        tool_call_id=None,
        store=None,
    )


def test_retrieve_embeds_once_then_searches_each_vector() -> None:
    """多个查询应在 Tool 内形成一次编码调用和多次独立向量搜索。"""

    context, embeddings, vector_store = _build_context()

    result = retrieve.func(
        ["问题一", "问题二", "问题三"],
        _build_tool_runtime(context),
    )

    assert embeddings.calls == [["问题一", "问题二", "问题三"]]
    assert vector_store.calls == [
        ([0.0, 1.0], 3),
        ([1.0, 2.0], 3),
        ([2.0, 3.0], 3),
    ]
    assert "查询 1：问题一" in result
    assert "查询 2：问题二" in result
    assert "查询 3：问题三" in result
    assert "文档ID：policy-1" in result
    assert "来源：财政部" in result
    assert "正文：" in result


def test_retrieve_tool_is_called_by_agent_with_runtime_context() -> None:
    context, embeddings, vector_store = _build_context()
    model = ToolCallingFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve",
                        "args": {"queries": ["公告是什么？", "发布时间是什么？"]},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="最终回答"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[retrieve],
        context_schema=RetrievalToolContext,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "公告是什么？"}]},
        context=context,
    )

    assert retrieve.name == "retrieve"
    assert result["messages"][-1].content == "最终回答"
    assert embeddings.calls == [["公告是什么？", "发布时间是什么？"]]
    assert len(vector_store.calls) == 2
    tool_result = next(
        message.content
        for message in result["messages"]
        if getattr(message, "name", None) == "retrieve"
    )
    assert "查询 1：公告是什么？" in tool_result
    assert "查询 2：发布时间是什么？" in tool_result


def test_retrieve_rejects_blank_query_before_embedding() -> None:
    context, embeddings, vector_store = _build_context()

    with pytest.raises(AppException) as exc_info:
        retrieve.func(
            ["有效问题", " "],
            _build_tool_runtime(context),
        )

    assert exc_info.value.stage == "validate_queries"
    assert embeddings.calls == []
    assert vector_store.calls == []
