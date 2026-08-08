"""验证 Runtime 直接创建和调用 LangChain 官方 Agent 的契约。"""

from unittest.mock import Mock

import pytest
from langchain.agents import create_agent as create_langchain_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
import src.service.runtime as runtime_module

from configuration import (
    CheckpointSettings,
    ContextCompressionSettings,
    IngestionSettings,
)
from src.errors import AppException, CHAT_AGENT_UNAVAILABLE
from src.service.agent.tools import (
    RetrievalToolContext,
    exact_paragraph_search,
    retrieve,
)
from src.service.agent.middlewares.title_generator import ChatTitleState, ConversationTitleGenerator
from src.service.runtime import Runtime


def test_runtime_creates_official_agent_directly(monkeypatch) -> None:
    """Runtime 直接读取配置并把参数传给官方 create_agent。"""

    captured: dict[str, object] = {}
    runnable = Mock()

    class FakeCheckpointConnection:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakePyMySQLSaver:
        def __init__(self, conn: object) -> None:
            self.conn = conn
            self.setup_called = False

        @staticmethod
        def parse_conn_string(conn_string: str) -> dict[str, object]:
            return {
                "host": "127.0.0.1",
                "user": "user",
                "password": "pass",
                "database": "chat_agent_checkpoint",
                "port": 3306,
                "unix_socket": None,
            }

        def setup(self) -> None:
            self.setup_called = True

    class FakeVectorStore:
        def get(self, *, include: list[str]) -> dict[str, list[str]]:
            assert include == []
            return {"ids": ["chunk-1"]}

    class FakeIngestion:
        def __init__(self, **kwargs: object) -> None:
            self.vector_store = FakeVectorStore()
            self.embeddings = Mock(name="batch_embeddings")

        def run(self, path: object, **kwargs: object) -> list[object]:
            raise AssertionError("已有 Chroma 数据时不应执行入库任务")

        def close(self) -> None:
            pass

    class FakeChatModel:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def create_agent(**kwargs: object):
        captured.update(kwargs)
        return runnable

    monkeypatch.setattr(runtime_module, "IngestionPipeline", FakeIngestion)
    monkeypatch.setattr(runtime_module, "ChatOpenAI", FakeChatModel)
    monkeypatch.setattr(runtime_module, "create_agent", create_agent)
    fake_conn = FakeCheckpointConnection()
    monkeypatch.setattr(runtime_module.pymysql, "connect", lambda **kwargs: fake_conn)
    monkeypatch.setattr(runtime_module, "PyMySQLSaver", FakePyMySQLSaver)

    runtime = Runtime(
        settings=IngestionSettings(),
        context_compression_settings=ContextCompressionSettings(enabled=False),
        checkpoint_settings=CheckpointSettings(
            database_url="mysql://user:pass@127.0.0.1:3306/chat_agent_checkpoint"
        ),
    )

    assert runtime.agent is runnable
    assert captured["tools"] == [exact_paragraph_search, retrieve]
    assert captured["context_schema"] is RetrievalToolContext
    assert captured["state_schema"] is ChatTitleState
    assert captured["middleware"] == [runtime.title_generator]
    assert isinstance(runtime.title_generator, ConversationTitleGenerator)
    assert captured["checkpointer"] is runtime.checkpointer
    assert isinstance(runtime.checkpointer, FakePyMySQLSaver)
    assert runtime.checkpointer.setup_called is True
    assert runtime._checkpoint_connection is fake_conn
    assert runtime.retrieval_context is not None
    assert runtime.ingestion is not None
    assert runtime.retrieval_context.vector_store is runtime.ingestion.vector_store
    assert runtime.retrieval_context.embeddings is runtime.ingestion.embeddings
    assert runtime.retrieval_context.top_k == 3
    runtime.close()
    assert runtime.checkpointer is None
    assert runtime._checkpoint_connection is None
    assert runtime.title_generator is None
    assert fake_conn.closed is True


def test_runtime_requires_checkpoint_database_url(monkeypatch) -> None:
    """启用 Checkpointer 但缺少数据库地址时必须显式失败。"""

    class FakeVectorStore:
        def get(self, *, include: list[str]) -> dict[str, list[str]]:
            assert include == []
            return {"ids": ["chunk-1"]}

    class FakeIngestion:
        def __init__(self, **kwargs: object) -> None:
            self.vector_store = FakeVectorStore()
            self.embeddings = Mock(name="batch_embeddings")

        def run(self, path: object, **kwargs: object) -> list[object]:
            raise AssertionError("已有 Chroma 数据时不应执行入库任务")

        def close(self) -> None:
            pass

    class FakeChatModel:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(runtime_module, "IngestionPipeline", FakeIngestion)
    monkeypatch.setattr(runtime_module, "ChatOpenAI", FakeChatModel)

    with pytest.raises(AppException) as exc_info:
        Runtime(
            settings=IngestionSettings(),
            checkpoint_settings=CheckpointSettings(database_url=None),
        )

    assert exc_info.value.error_code is CHAT_AGENT_UNAVAILABLE


def test_runtime_invokes_official_agent_with_thread_id() -> None:
    """Runtime 直接调用官方 Agent，并把 thread_id 传给 checkpointer。"""

    runnable = Mock()
    runnable.invoke.return_value = {"messages": [AIMessage(content="回答")]}
    runtime = Runtime.__new__(Runtime)
    runtime.retrieval_context = RetrievalToolContext(
        vector_store=Mock(),
        embeddings=Mock(),
        top_k=3,
    )
    runtime.agent = runnable
    runtime.checkpointer = Mock()
    runtime._observation_enabled = False

    assert runtime.answer("问题", thread_id="thread-1") == "回答"
    assert runnable.invoke.call_args.kwargs["config"] == {
        "configurable": {"thread_id": "thread-1"}
    }
    context = runnable.invoke.call_args.kwargs["context"]
    assert isinstance(context, RetrievalToolContext)
    assert context is runtime.retrieval_context
    assert runnable.invoke.call_args.args[0]["messages"][0].content == "问题"


def test_runtime_answer_with_title_returns_generated_title() -> None:
    """answer_with_title 只在本轮真正生成了标题时返回标题。"""

    runnable = Mock()
    runnable.invoke.return_value = {
        "messages": [AIMessage(content="回答")],
        "generated_title": "测试标题",
        "title_generated": True,
    }
    runtime = Runtime.__new__(Runtime)
    runtime.retrieval_context = RetrievalToolContext(
        vector_store=Mock(),
        embeddings=Mock(),
        top_k=3,
    )
    runtime.agent = runnable
    runtime.checkpointer = Mock()
    runtime._observation_enabled = False

    answer, title = runtime.answer_with_title("问题", thread_id="thread-1")
    assert answer == "回答"
    assert title == "测试标题"

    runnable.invoke.return_value = {
        "messages": [AIMessage(content="回答")],
        "generated_title": "测试标题",
        "title_generated": False,
    }
    answer, title = runtime.answer_with_title("问题", thread_id="thread-1")
    assert answer == "回答"
    assert title is None


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
