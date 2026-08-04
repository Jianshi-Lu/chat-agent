"""验证 Runtime 直接创建和调用 LangChain 官方 Agent 的契约。"""

from unittest.mock import Mock

import pytest
from langchain.agents import create_agent as create_langchain_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
import src.service.runtime as runtime_module

from configuration import ContextCompressionSettings, KnowledgeRuntimeSettings
from src.service.agent.tools import exact_paragraph_search
from src.service.runtime import Runtime


def test_runtime_creates_official_agent_directly(monkeypatch) -> None:
    """Runtime 直接读取配置并把参数传给官方 create_agent。"""

    captured: dict[str, object] = {}
    retriever_tool = Mock()
    runnable = Mock()

    class FakeCollection:
        def count(self) -> int:
            return 1

    class FakeIngestion:
        policy_chunk_collection = FakeCollection()

        def __init__(self, **kwargs: object) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeRetriever:
        def __init__(self, ingestion: FakeIngestion) -> None:
            pass

    class FakeChatModel:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def create_agent(**kwargs: object):
        captured.update(kwargs)
        return runnable

    monkeypatch.setattr(runtime_module, "IngestionService", FakeIngestion)
    monkeypatch.setattr(runtime_module, "Retriever", FakeRetriever)
    monkeypatch.setattr(
        runtime_module,
        "build_retriever_tool",
        lambda retriever, top_k: retriever_tool,
    )
    monkeypatch.setattr(runtime_module, "ChatOpenAI", FakeChatModel)
    monkeypatch.setattr(runtime_module, "create_agent", create_agent)

    runtime = Runtime(
        settings=KnowledgeRuntimeSettings(),
        context_compression_settings=ContextCompressionSettings(enabled=False),
    )

    assert runtime.agent is runnable
    assert captured["tools"] == [exact_paragraph_search, retriever_tool]
    assert captured["middleware"] == []
    assert captured["checkpointer"] is None
    runtime.close()


def test_runtime_invokes_official_agent_with_thread_id() -> None:
    """Runtime 直接调用官方 Agent，并把 thread_id 传给 checkpointer。"""

    runnable = Mock()
    runnable.invoke.return_value = {"messages": [AIMessage(content="回答")]}
    runtime = Runtime.__new__(Runtime)
    runtime.retriever = Mock()
    runtime.retriever_tool = Mock()
    runtime.agent = runnable
    runtime.checkpointer = Mock()

    assert runtime.answer("问题", thread_id="thread-1") == "回答"
    assert runnable.invoke.call_args.kwargs["config"] == {
        "configurable": {"thread_id": "thread-1"}
    }
    assert runnable.invoke.call_args.args[0]["messages"][0].content == "问题"


def test_official_summarization_persists_compacted_state() -> None:
    """官方 Middleware 应在模型调用前摘要旧消息并写回 Checkpoint。"""

    summary_model = FakeMessagesListChatModel(
        responses=[AIMessage(content="历史摘要")]
    )
    main_model = FakeMessagesListChatModel(
        responses=[AIMessage(content="当前回答")]
    )
    checkpointer = InMemorySaver()
    runnable = create_langchain_agent(
        model=main_model,
        middleware=[
            SummarizationMiddleware(
                model=summary_model,
                trigger=("messages", 2),
                keep=("messages", 1),
            )
        ],
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "compression-test"}}

    result = runnable.invoke(
        {
            "messages": [
                HumanMessage(content="问题一"),
                AIMessage(content="回答一"),
                HumanMessage(content="问题二"),
            ]
        },
        config=config,
    )
    state = runnable.get_state(config)
    state_messages = state.values["messages"]

    assert result["messages"][-1].content == "当前回答"
    assert any("历史摘要" in message.content for message in state_messages)
    assert any(message.content == "问题二" for message in state_messages)


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"messages": []},
        {"messages": [HumanMessage(content="不是模型回答")]},
        {"messages": [AIMessage(content="")]},
    ],
)
def test_runtime_rejects_invalid_agent_result(result: object) -> None:
    """非法官方 Agent 状态必须显式失败，不能静默返回空字符串。"""

    with pytest.raises((TypeError, ValueError)):
        Runtime._result_text(result)
