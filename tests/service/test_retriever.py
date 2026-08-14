"""验证 Agent 检索 Tool 只编排独立 vector-service 客户端。"""

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from src.errors import AppException, KNOWLEDGE_RETRIEVAL_FAILED
from src.service.agent.tools.retrieve import (
    RetrievalToolContext,
    find_file_by_name,
    get_bid_record,
    list_files,
    list_bid_records,
    read_credential_file,
    read_knowledge_file,
    retrieve_bid_records,
    retrieve_credential_files_by_bid_record_id,
    retrieve_credential_files,
    retrieve_knowledge_files,
)


class RecordingVectorClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def search_knowledge(self, queries: list[str]) -> str:
        self.calls.append(("knowledge", list(queries)))
        return "公共知识检索结果"

    def list_files(self) -> str:
        self.calls.append(("list_files", None))
        return "已入库文件列表"

    def find_file_by_name(self, filename: str) -> str:
        self.calls.append(("find_file_by_name", filename))
        return (
            "类型：知识文件；文件名：complex_pdf (1).pdf；"
            "document_uuid：550e8400-e29b-41d4-a716-446655440000"
        )

    def read_knowledge_file(self, document_uuid: str) -> str:
        self.calls.append(("read_knowledge_file", document_uuid))
        return "知识文件完整正文"

    def read_credential_file(self, file_uuid: str) -> str:
        self.calls.append(("read_credential_file", file_uuid))
        return "资信文件正文和关联投标记录"

    def search_credential_files(self, queries: list[str]) -> str:
        self.calls.append(("credential_files", list(queries)))
        return "跨投标记录检索结果"

    def search_bid_records(self, queries: list[str]) -> str:
        self.calls.append(("bid_records", list(queries)))
        return "跨投标记录结构化检索结果"

    def search_credential_files_by_bid_record_id(
        self,
        bid_record_id: int,
        queries: list[str],
    ) -> str:
        self.calls.append(("scoped_credential_files", (bid_record_id, list(queries))))
        return "指定投标记录资信检索结果"

    def get_bid_record(self, bid_record_id: int) -> str:
        self.calls.append(("get_bid_record", bid_record_id))
        return f"投标记录 {bid_record_id}"

    def list_bid_records(self) -> str:
        self.calls.append(("list_bid_records", None))
        return "已入库投标记录"


class ToolCallingFakeChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _build_context() -> tuple[RetrievalToolContext, RecordingVectorClient]:
    client = RecordingVectorClient()
    return RetrievalToolContext(vector_client=client), client


def _build_tool_runtime(context: RetrievalToolContext) -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=lambda _: None,
        tool_call_id=None,
        store=None,
    )


def test_retrieve_knowledge_files_normalizes_queries_and_delegates_once() -> None:
    context, client = _build_context()

    result = retrieve_knowledge_files.func(
        [" 问题一 ", "问题二"],
        _build_tool_runtime(context),
    )

    assert result == "公共知识检索结果"
    assert client.calls == [("knowledge", ["问题一", "问题二"])]


def test_retrieve_knowledge_files_tool_is_called_by_agent_with_runtime_context() -> None:
    context, client = _build_context()
    model = ToolCallingFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve_knowledge_files",
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
        tools=[retrieve_knowledge_files],
        context_schema=RetrievalToolContext,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "公告是什么？"}]},
        context=context,
    )

    assert result["messages"][-1].content == "最终回答"
    assert client.calls == [
        ("knowledge", ["公告是什么？", "发布时间是什么？"])
    ]


def test_no_argument_tool_is_called_by_agent_with_runtime_context() -> None:
    context, client = _build_context()
    model = ToolCallingFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_bid_records",
                        "args": {},
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
        tools=[list_bid_records],
        context_schema=RetrievalToolContext,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "列出投标记录"}]},
        context=context,
    )

    assert result["messages"][-1].content == "最终回答"
    assert client.calls == [("list_bid_records", None)]


def test_filename_tools_are_called_by_agent_in_sequence() -> None:
    context, client = _build_context()
    document_uuid = "550e8400-e29b-41d4-a716-446655440000"
    model = ToolCallingFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_file_by_name",
                        "args": {"filename": "complex_pdf (1).pdf"},
                        "id": "call-find",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_knowledge_file",
                        "args": {"document_uuid": document_uuid},
                        "id": "call-read",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="该文件包含三页测试内容。"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[find_file_by_name, read_knowledge_file],
        context_schema=RetrievalToolContext,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "complex_pdf (1).pdf 讲讲这个文件的内容",
                }
            ]
        },
        context=context,
    )

    assert result["messages"][-1].content == "该文件包含三页测试内容。"
    assert client.calls == [
        ("find_file_by_name", "complex_pdf (1).pdf"),
        ("read_knowledge_file", document_uuid),
    ]


def test_retrieve_knowledge_files_rejects_blank_query_before_http_call() -> None:
    context, client = _build_context()

    with pytest.raises(AppException) as exc_info:
        retrieve_knowledge_files.func(
            ["有效问题", " "],
            _build_tool_runtime(context),
        )

    assert exc_info.value.stage == "validate_queries"
    assert client.calls == []


def test_private_tools_keep_dynamic_scope_and_exact_record_id() -> None:
    context, client = _build_context()
    runtime = _build_tool_runtime(context)

    assert retrieve_credential_files.func(["甲级资质"], runtime) == "跨投标记录检索结果"
    assert (
        retrieve_bid_records.func(["报价"], runtime)
        == "跨投标记录结构化检索结果"
    )
    assert (
        retrieve_credential_files_by_bid_record_id.func(13, ["资质等级"], runtime)
        == "指定投标记录资信检索结果"
    )
    assert get_bid_record.func(13, runtime) == "投标记录 13"
    assert list_bid_records.func(runtime) == "已入库投标记录"
    assert client.calls == [
        ("credential_files", ["甲级资质"]),
        ("bid_records", ["报价"]),
        ("scoped_credential_files", (13, ["资质等级"])),
        ("get_bid_record", 13),
        ("list_bid_records", None),
    ]


def test_file_tools_delegate_with_stable_identifiers() -> None:
    context, client = _build_context()
    runtime = _build_tool_runtime(context)
    document_uuid = "550e8400-e29b-41d4-a716-446655440000"
    file_uuid = "550e8400-e29b-41d4-a716-446655440001"

    assert list_files.func(runtime) == "已入库文件列表"
    assert (
        "complex_pdf (1).pdf"
        in find_file_by_name.func(" complex_pdf (1).pdf ", runtime)
    )
    assert read_knowledge_file.func(document_uuid, runtime) == "知识文件完整正文"
    assert (
        read_credential_file.func(file_uuid, runtime)
        == "资信文件正文和关联投标记录"
    )
    assert client.calls == [
        ("list_files", None),
        ("find_file_by_name", "complex_pdf (1).pdf"),
        ("read_knowledge_file", document_uuid),
        ("read_credential_file", file_uuid),
    ]


def test_tool_preserves_structured_vector_service_error() -> None:
    class FailingClient(RecordingVectorClient):
        def search_knowledge(self, queries: list[str]) -> str:
            raise AppException(
                KNOWLEDGE_RETRIEVAL_FAILED,
                stage="connect_vector_service",
            )

    context = RetrievalToolContext(vector_client=FailingClient())
    with pytest.raises(AppException) as exc_info:
        retrieve_knowledge_files.func(["问题"], _build_tool_runtime(context))

    assert exc_info.value.stage == "connect_vector_service"
