"""验证悬挂历史只从模型输入隐藏，原始 Checkpoint 仍可恢复。"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from typing_extensions import override

from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from src.service.agent.middlewares.history_projection import (
    CheckpointHistoryProjectionMiddleware,
    ProjectedSummarizationMiddleware,
    project_messages_for_model,
)


class RecordingMessagesModel(FakeMessagesListChatModel):
    """记录模型实际收到的消息，用于断言投影边界。"""

    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _tool_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "list_bid_records",
                "args": {},
                "id": "old-tool-call",
                "type": "tool_call",
            }
        ],
    )


def _contents(messages: list[BaseMessage]) -> list[str]:
    return [str(message.content) for message in messages]


def test_projection_hides_only_an_unfinished_historical_turn() -> None:
    messages = [
        HumanMessage(content="已完成的问题"),
        AIMessage(content="已完成的回答"),
        SystemMessage(content="旧轮取证"),
        HumanMessage(content="列出所有投标记录"),
        _tool_call(),
        ToolMessage(
            name="list_bid_records",
            tool_call_id="old-tool-call",
            content="投标记录 10、13、18",
        ),
        SystemMessage(content="本轮投标记录 25 的取证"),
        HumanMessage(content="查询投标记录 25 的基本信息和资信材料"),
    ]

    projected = project_messages_for_model(messages)
    contents = _contents(projected)

    assert "已完成的问题" in contents
    assert "已完成的回答" in contents
    assert "列出所有投标记录" not in contents
    assert "投标记录 10、13、18" not in contents
    assert "本轮投标记录 25 的取证" in contents
    assert "查询投标记录 25 的基本信息和资信材料" in contents
    assert _contents(messages)[3] == "列出所有投标记录"


def test_projection_keeps_completed_empty_tool_result_and_current_tool_progress() -> None:
    messages = [
        HumanMessage(content="查找不存在的资信文件"),
        _tool_call(),
        ToolMessage(
            name="list_bid_records",
            tool_call_id="old-tool-call",
            content="[]",
        ),
        AIMessage(content="未找到对应的资信文件。"),
        HumanMessage(content="查询投标记录 25"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_bid_record",
                    "args": {"bid_record_id": 25},
                    "id": "current-tool-call",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            name="get_bid_record",
            tool_call_id="current-tool-call",
            content="投标记录 ID：25",
        ),
    ]

    projected = project_messages_for_model(messages)

    assert _contents(projected) == _contents(messages)


def test_agent_hides_dangling_history_from_model_and_keeps_checkpoint_state() -> None:
    model = RecordingMessagesModel(
        responses=[AIMessage(content="初始化回答"), AIMessage(content="仅回答投标记录 25")]
    )
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        middleware=[CheckpointHistoryProjectionMiddleware()],
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "projection-with-checkpoint"}}

    agent.invoke({"messages": [HumanMessage(content="初始化问题")]}, config=config)
    agent.update_state(
        config,
        {
            "messages": [
                HumanMessage(content="列出所有投标记录"),
                _tool_call(),
                ToolMessage(
                    name="list_bid_records",
                    tool_call_id="old-tool-call",
                    content="投标记录 10、13、18",
                ),
            ]
        },
    )

    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content="本轮仅有投标记录 25 的证据"),
                HumanMessage(content="查询投标记录 25"),
            ]
        },
        config=config,
    )

    seen_contents = _contents(model.seen_messages[-1])
    checkpoint_contents = _contents(agent.get_state(config).values["messages"])

    assert result["messages"][-1].content == "仅回答投标记录 25"
    assert "列出所有投标记录" not in seen_contents
    assert "投标记录 10、13、18" not in seen_contents
    assert "本轮仅有投标记录 25 的证据" in seen_contents
    assert "查询投标记录 25" in seen_contents
    assert "列出所有投标记录" in checkpoint_contents
    assert "投标记录 10、13、18" in checkpoint_contents


def test_projected_summary_excludes_dangling_history_but_history_snapshots_keep_it() -> None:
    main_model = RecordingMessagesModel(
        responses=[AIMessage(content="初始化回答"), AIMessage(content="仅回答投标记录 25")]
    )
    summary_model = RecordingMessagesModel(responses=[AIMessage(content="干净的历史摘要")])
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=main_model,
        middleware=[
            CheckpointHistoryProjectionMiddleware(),
            ProjectedSummarizationMiddleware(
                model=summary_model,
                trigger=("messages", 3),
                keep=("messages", 1),
            ),
        ],
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "projection-with-summary"}}

    agent.invoke({"messages": [HumanMessage(content="初始化问题")]}, config=config)
    agent.update_state(
        config,
        {
            "messages": [
                HumanMessage(content="列出所有投标记录"),
                _tool_call(),
                ToolMessage(
                    name="list_bid_records",
                    tool_call_id="old-tool-call",
                    content="投标记录 10、13、18",
                ),
            ]
        },
    )

    agent.invoke(
        {"messages": [HumanMessage(content="查询投标记录 25")]},
        config=config,
    )

    summary_prompt = "\n".join(_contents(summary_model.seen_messages[-1]))
    main_input = _contents(main_model.seen_messages[-1])
    history_contents = [
        _contents(snapshot.values["messages"])
        for snapshot in agent.get_state_history(config)
    ]

    assert "列出所有投标记录" not in summary_prompt
    assert "投标记录 10、13、18" not in summary_prompt
    assert "列出所有投标记录" not in main_input
    assert "投标记录 10、13、18" not in main_input
    assert any("列出所有投标记录" in contents for contents in history_contents)
