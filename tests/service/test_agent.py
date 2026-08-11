"""验证 Runtime 直接创建和调用 LangChain 官方 Agent 的契约。"""

from unittest.mock import Mock

import pytest
from langchain.agents import create_agent as create_langchain_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import src.service.runtime as runtime_module
from configuration import (
    CheckpointSettings,
    ContextCompressionSettings,
    VectorServiceSettings,
)
from src.errors import AppException, CHAT_AGENT_UNAVAILABLE
from src.service.agent.middlewares.title_generator import (
    ChatTitleState,
    ConversationTitleGenerator,
)
from src.service.agent.tools import (
    RetrievalToolContext,
    find_ingested_files,
    get_bid_record,
    list_ingested_files,
    list_bid_records,
    read_credential_file,
    read_knowledge_file,
    retrieve,
    retrieve_bid_credentials,
    retrieve_bid_materials,
)
from src.service.runtime import Runtime


class FakeVectorClient:
    instances: list["FakeVectorClient"] = []

    def __init__(self, settings: VectorServiceSettings) -> None:
        self.settings = settings
        self.health_called = False
        self.closed = False
        self.instances.append(self)

    def health(self) -> None:
        self.health_called = True

    def close(self) -> None:
        self.closed = True


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


class FakeChatModel:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_runtime_creates_official_agent_with_vector_service_context(monkeypatch) -> None:
    captured: dict[str, object] = {}
    runnable = Mock()
    FakeVectorClient.instances.clear()

    def create_agent(**kwargs: object):
        captured.update(kwargs)
        return runnable

    monkeypatch.setattr(runtime_module, "VectorServiceClient", FakeVectorClient)
    monkeypatch.setattr(runtime_module, "ChatOpenAI", FakeChatModel)
    monkeypatch.setattr(runtime_module, "create_agent", create_agent)
    fake_conn = FakeCheckpointConnection()
    monkeypatch.setattr(runtime_module.pymysql, "connect", lambda **kwargs: fake_conn)
    monkeypatch.setattr(runtime_module, "PyMySQLSaver", FakePyMySQLSaver)

    runtime = Runtime(
        vector_service_settings=VectorServiceSettings(),
        context_compression_settings=ContextCompressionSettings(enabled=False),
        checkpoint_settings=CheckpointSettings(
            database_url="mysql://user:pass@127.0.0.1:3306/chat_agent_checkpoint"
        ),
    )

    assert runtime.agent is runnable
    assert captured["tools"] == [
        retrieve,
        list_ingested_files,
        find_ingested_files,
        read_knowledge_file,
        read_credential_file,
        list_bid_records,
        get_bid_record,
        retrieve_bid_materials,
        retrieve_bid_credentials,
    ]
    assert captured["context_schema"] is RetrievalToolContext
    assert captured["state_schema"] is ChatTitleState
    assert captured["middleware"] == [runtime.title_generator]
    assert isinstance(runtime.title_generator, ConversationTitleGenerator)
    assert captured["checkpointer"] is runtime.checkpointer
    assert isinstance(runtime.checkpointer, FakePyMySQLSaver)
    assert runtime.checkpointer.setup_called is True
    assert runtime.retrieval_context is not None
    assert runtime.retrieval_context.vector_client is runtime.vector_client
    assert FakeVectorClient.instances[0].health_called is True

    runtime.close()
    assert fake_conn.closed is True
    assert FakeVectorClient.instances[0].closed is True


def test_runtime_requires_checkpoint_database_url(monkeypatch) -> None:
    FakeVectorClient.instances.clear()
    monkeypatch.setattr(runtime_module, "VectorServiceClient", FakeVectorClient)
    monkeypatch.setattr(runtime_module, "ChatOpenAI", FakeChatModel)

    with pytest.raises(AppException) as exc_info:
        Runtime(checkpoint_settings=CheckpointSettings(database_url=None))

    assert exc_info.value.error_code is CHAT_AGENT_UNAVAILABLE
    assert FakeVectorClient.instances[0].closed is True


def _runtime_with_runnable(runnable: Mock) -> Runtime:
    runtime = Runtime.__new__(Runtime)
    vector_client = Mock()
    vector_client.search_knowledge.return_value = "正文：PUBLIC-E2E-8047"
    vector_client.get_bid_record.return_value = "投标记录 ID：13\n投标人名称：testing"
    vector_client.search_bid_credentials.return_value = (
        "投标记录 ID：13\n文件：credential.pdf"
    )
    runtime.retrieval_context = RetrievalToolContext(vector_client=vector_client)
    runtime.agent = runnable
    runtime.checkpointer = Mock()
    runtime._observation_enabled = False
    return runtime


def test_runtime_invokes_official_agent_with_thread_id() -> None:
    runnable = Mock()
    runnable.invoke.return_value = {"messages": [AIMessage(content="回答")]}
    runtime = _runtime_with_runnable(runnable)

    assert runtime.answer("问题", thread_id="thread-1") == "回答"
    assert runnable.invoke.call_args.kwargs["config"] == {
        "configurable": {"thread_id": "thread-1"},
        "recursion_limit": 25,
    }
    assert runnable.invoke.call_args.kwargs["context"] is runtime.retrieval_context
    assert runnable.invoke.call_args.args[0]["messages"][0].content == "问题"


def test_runtime_answer_with_title_returns_only_new_title() -> None:
    runnable = Mock()
    runnable.invoke.return_value = {
        "messages": [AIMessage(content="回答")],
        "generated_title": "测试标题",
        "title_generated": True,
    }
    runtime = _runtime_with_runnable(runnable)

    assert runtime.answer_with_title("问题", thread_id="thread-1") == (
        "回答",
        "测试标题",
    )
    runnable.invoke.return_value["title_generated"] = False
    assert runtime.answer_with_title("问题", thread_id="thread-1") == (
        "回答",
        None,
    )


def test_runtime_returns_only_ids_referenced_by_material_tools() -> None:
    runnable = Mock()
    runnable.invoke.return_value = {
        "messages": [
            ToolMessage(
                name="retrieve_bid_credentials",
                tool_call_id="tool-1",
                content=(
                    "查询 1：证书编号\n投标记录 ID：17\n"
                    "文件：credential.pdf"
                ),
            ),
            ToolMessage(
                name="list_bid_records",
                tool_call_id="tool-2",
                content="已入库投标记录：\nID 99；投标人 不应作为下载引用",
            ),
            AIMessage(content="投标记录 17 的证书编号已查到。"),
        ]
    }
    runtime = _runtime_with_runnable(runnable)

    result = runtime.answer_with_references("查询资信材料", thread_id="thread-1")

    assert result.answer == "投标记录 17 的证书编号已查到。"
    assert result.referenced_bid_record_ids == (17,)


def test_runtime_keeps_only_the_explicitly_requested_bid_record_reference() -> None:
    runnable = Mock()
    runnable.invoke.return_value = {
        "messages": [
            ToolMessage(
                name="retrieve_bid_materials",
                tool_call_id="tool-1",
                content="投标记录 ID：13\n投标记录 ID：11",
            ),
            AIMessage(content="已查询。"),
        ]
    }
    runtime = _runtime_with_runnable(runnable)

    result = runtime.answer_with_references(
        "投标记录 13 的资信材料有哪些问题？",
        thread_id="thread-1",
    )

    assert result.referenced_bid_record_ids == (13,)


def test_runtime_retrieves_current_bid_credentials_before_reusing_history() -> None:
    question = "投标记录 13 的资信材料有哪些问题？"
    runnable = Mock()
    runnable.invoke.return_value = {
        "messages": [
            HumanMessage(content=question),
            AIMessage(content="当前资信材料已核查。"),
        ]
    }
    runtime = _runtime_with_runnable(runnable)
    runtime.retrieval_context.vector_client.get_bid_record.return_value = (
        "投标记录 ID：13\n投标人名称：testing"
    )
    runtime.retrieval_context.vector_client.search_bid_credentials.return_value = (
        "投标记录 ID：13\n文件：complex_pdf.pdf"
    )

    result = runtime.answer_with_references(question, thread_id="thread-1")

    assert result.referenced_bid_record_ids == (13,)
    runtime.retrieval_context.vector_client.get_bid_record.assert_called_once_with(13)
    runtime.retrieval_context.vector_client.search_bid_credentials.assert_called_once_with(
        13,
        [question],
    )
    invoke_messages = runnable.invoke.call_args.args[0]["messages"]
    assert isinstance(invoke_messages[0], SystemMessage)
    assert "complex_pdf.pdf" in invoke_messages[0].content


def test_runtime_does_not_reuse_deleted_bid_record_from_history() -> None:
    question = "投标记录 20 的资信材料有哪些问题？"
    runnable = Mock()
    runtime = _runtime_with_runnable(runnable)
    runtime.retrieval_context.vector_client.get_bid_record.return_value = (
        "未找到投标记录 20。"
    )

    result = runtime.answer_with_references(question, thread_id="thread-1")

    assert result.answer == (
        "当前投标资料中未找到投标记录 ID：20，"
        "无法依据已删除或不存在的记录提供投标人、报价或资信材料。"
    )
    assert result.referenced_bid_record_ids == ()
    runnable.invoke.assert_not_called()
    runtime.retrieval_context.vector_client.search_bid_credentials.assert_not_called()


def test_runtime_never_attaches_bid_file_cards_to_a_public_knowledge_identifier() -> None:
    runnable = Mock()
    runnable.invoke.return_value = {
        "messages": [
            ToolMessage(
                name="retrieve_bid_materials",
                tool_call_id="tool-1",
                content="投标记录 ID：13",
            ),
            AIMessage(content="已查询。"),
        ]
    }
    runtime = _runtime_with_runnable(runnable)

    result = runtime.answer_with_references(
        "PUBLIC-E2E-8047 的投标保证金是多少？",
        thread_id="thread-1",
    )

    assert result.referenced_bid_record_ids == ()


def test_runtime_rejects_public_identifier_answer_without_tool_evidence() -> None:
    runnable = Mock()
    runnable.invoke.return_value = {
        "messages": [
            ToolMessage(
                name="retrieve",
                tool_call_id="tool-1",
                content="文件：通用说明.pdf\n正文：保证金说明",
            ),
            AIMessage(content="保证金是 2,000,000 元。"),
        ]
    }
    runtime = _runtime_with_runnable(runnable)
    runtime.retrieval_context.vector_client.search_knowledge.return_value = (
        "未找到包含公共知识编号“PUBLIC-E2E-8047”的当前已入库内容。"
    )

    result = runtime.answer_with_references(
        "PUBLIC-E2E-8047 的保证金是多少？",
        thread_id="thread-1",
    )

    assert result.answer == (
        "当前已入库公共知识中未检索到编号“PUBLIC-E2E-8047”，"
        "无法依据现有资料给出该编号的具体答案。"
    )


def test_runtime_keeps_public_identifier_answer_with_tool_evidence() -> None:
    runnable = Mock()
    runnable.invoke.return_value = {
        "messages": [
            ToolMessage(
                name="retrieve",
                tool_call_id="tool-1",
                content="正文：PUBLIC-E2E-8047，保证金 88,888 元",
            ),
            AIMessage(content="保证金是 88,888 元。"),
        ]
    }
    runtime = _runtime_with_runnable(runnable)

    result = runtime.answer_with_references(
        "PUBLIC-E2E-8047 的保证金是多少？",
        thread_id="thread-1",
    )

    assert result.answer == "保证金是 88,888 元。"


def test_runtime_does_not_reuse_previous_turn_tool_evidence() -> None:
    question = "PUBLIC-E2E-LIVE-1527 的保证金是多少？"
    runnable = Mock()
    runnable.invoke.return_value = {
        "messages": [
            HumanMessage(content=question),
            ToolMessage(
                name="retrieve",
                tool_call_id="old-tool",
                content="正文：PUBLIC-E2E-LIVE-1527，保证金 152,700 元",
            ),
            AIMessage(content="保证金是 152,700 元。"),
            HumanMessage(content=question),
            ToolMessage(
                name="retrieve",
                tool_call_id="current-tool",
                content="文件：通用说明.pdf\n正文：保证金说明",
            ),
            ToolMessage(
                name="retrieve_bid_materials",
                tool_call_id="current-material-tool",
                content="投标记录 ID：13",
            ),
            AIMessage(content="保证金是 152,700 元。"),
        ]
    }
    answer = Runtime._ground_public_knowledge_answer(
        "保证金是 152,700 元。",
        runnable.invoke.return_value,
        question,
    )
    references = Runtime._referenced_bid_record_ids(
        runnable.invoke.return_value,
        question=question,
    )

    assert answer == (
        "当前已入库公共知识中未检索到编号“PUBLIC-E2E-LIVE-1527”，"
        "无法依据现有资料给出该编号的具体答案。"
    )
    assert references == ()


def test_runtime_retrieves_current_public_evidence_before_reusing_history() -> None:
    question = "PUBLIC-E2E-LIVE-1527 的保证金是多少？"
    runnable = Mock()
    runnable.invoke.return_value = {
        "messages": [
            SystemMessage(content="本轮证据"),
            HumanMessage(content=question),
            AIMessage(content="保证金是 152,700 元。"),
        ]
    }
    runtime = _runtime_with_runnable(runnable)
    runtime.retrieval_context.vector_client.search_knowledge.return_value = (
        "正文：PUBLIC-E2E-LIVE-1527，保证金 152,700 元"
    )

    result = runtime.answer_with_references(question, thread_id="thread-1")

    assert result.answer == "保证金是 152,700 元。"
    runtime.retrieval_context.vector_client.search_knowledge.assert_called_once_with(
        [question]
    )
    invoke_messages = runnable.invoke.call_args.args[0]["messages"]
    assert isinstance(invoke_messages[0], SystemMessage)
    assert "PUBLIC-E2E-LIVE-1527" in invoke_messages[0].content


def test_runtime_hides_internal_file_identity_from_user_answer() -> None:
    runnable = Mock()
    runnable.invoke.return_value = {
        "messages": [
            AIMessage(
                content=(
                    "- **资信文件名**：`credential.pdf`\n"
                    "- **文件 UUID**：`550e8400-e29b-41d4-a716-446655440000`\n"
                    "- **可下载路径**：bid-records/17/converted/credential.md\n"
                    "根据政策文件（文件 ID：550e8400-e29b-41d4-a716-446655440000）说明。\n"
                    "该资信文件（资信文件 ID：40，）需要下载。\n"
                    "**依据来源**：credential.pdf（file_uuid: "
                    "`550e8400-e29b-41d4-a716-446655440000`，资信文件 ID 40）。\n"
                    "请使用界面中的下载按钮。"
                )
            )
        ]
    }
    runtime = _runtime_with_runnable(runnable)

    result = runtime.answer_with_references("查询资信材料", thread_id="thread-1")

    assert result.answer == (
        "- **资信文件名**：`credential.pdf`\n"
        "根据政策文件说明。\n"
        "该资信文件需要下载。\n"
        "**依据来源**：credential.pdf。\n"
        "请使用界面中的下载按钮。"
    )


def test_runtime_context_is_not_prebound_to_a_bid_record() -> None:
    runnable = Mock()
    runnable.invoke.return_value = {"messages": [AIMessage(content="回答")]}
    runtime = _runtime_with_runnable(runnable)

    runtime.answer("资质是什么？", thread_id="thread-dynamic")

    context = runnable.invoke.call_args.kwargs["context"]
    assert context is runtime.retrieval_context
    assert not hasattr(context, "bid_context")


def test_official_summarization_persists_compacted_state() -> None:
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
    state_messages = runnable.get_state(config).values["messages"]

    assert result["messages"][-1].content == "当前回答"
    assert any("历史摘要" in message.content for message in state_messages)
    assert any(message.content == "问题二" for message in state_messages)
