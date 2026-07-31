"""验证 LangChain 官方 Agent 的无状态调用和流式行为。"""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from app.agent import LangChainChatAgent


def build_agent(monkeypatch):
    """替换官方 Agent 工厂，验证入口只传递当前消息。"""

    calls: list[list[BaseMessage]] = []
    agent = Mock()

    def invoke(inputs: dict[str, list[BaseMessage]]) -> dict[str, list[BaseMessage]]:
        messages = inputs["messages"]
        calls.append(list(messages))
        return {"messages": [*messages, AIMessage(content=f"回答{len(calls)}")]}

    def stream(
        inputs: dict[str, list[BaseMessage]],
        *,
        stream_mode: str,
        version: str,
    ):
        messages = inputs["messages"]
        calls.append(list(messages))
        return iter(
            [
                {
                    "type": "messages",
                    "data": (
                        AIMessageChunk(content="工具节点文本"),
                        {"langgraph_node": "tools"},
                    ),
                },
                {
                    "type": "messages",
                    "data": (
                        AIMessage(
                            content="工具决策文本",
                            tool_calls=[
                                {
                                    "name": "exact_paragraph_search",
                                    "args": {
                                        "chapter_no": 3,
                                        "paragraph_no": 2,
                                    },
                                    "id": "call-1",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        {"langgraph_node": "model"},
                    ),
                },
                {
                    "type": "messages",
                    "data": (
                        AIMessageChunk(content="流"),
                        {"langgraph_node": "model"},
                    ),
                },
                {
                    "type": "messages",
                    "data": (
                        AIMessageChunk(content="式回答"),
                        {"langgraph_node": "model"},
                    ),
                },
            ]
        )

    agent.invoke.side_effect = invoke
    agent.stream.side_effect = stream
    monkeypatch.setattr("app.agent.agent.create_agent", lambda **kwargs: agent)
    return LangChainChatAgent(), calls


def test_agent_does_not_preserve_context(monkeypatch) -> None:
    """同一 Agent 实例的每次调用都只能看到当前消息。"""

    agent, calls = build_agent(monkeypatch)

    assert agent.complete("第一问") == "回答1"
    assert agent.complete("第二问") == "回答2"

    assert [message.content for message in calls[0]] == ["第一问"]
    assert [message.content for message in calls[1]] == ["第二问"]


def test_agent_streams_without_saving_context(monkeypatch) -> None:
    """流式片段应按顺序返回，后续调用不应包含上一轮内容。"""

    agent, calls = build_agent(monkeypatch)

    assert list(agent.stream_response("流式问题")) == [
        "流",
        "式回答",
    ]
    assert agent.complete("下一问") == "回答2"

    assert [message.content for message in calls[1]] == ["下一问"]


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"messages": []},
        {"messages": [BaseMessage(content="不是模型回答", type="base")]},
        {"messages": [AIMessage(content="")]},
    ],
)
def test_agent_rejects_invalid_result(result: object) -> None:
    """非法 Agent 状态必须显式失败，不能静默返回空字符串。"""

    with pytest.raises((TypeError, ValueError)):
        LangChainChatAgent._result_text(result)


def test_client_creates_default_langchain_agent() -> None:
    """省略 Agent 时，ChatClient 应能根据配置构造本地模型客户端。"""

    from app.agent import LangChainChatAgent
    from app.service.client import ChatClient

    client = ChatClient()

    assert isinstance(client._agent, LangChainChatAgent)
