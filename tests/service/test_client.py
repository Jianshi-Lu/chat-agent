"""验证 ChatClient 的 Agent 转发和流式边界。"""

from dataclasses import dataclass, field

import pytest

from app.service.client import ChatClient
from app.errors import AppException, CHAT_AGENT_FAILED


@dataclass
class FakeAgent:
    """记录 ChatClient 转发请求的测试 Agent。"""

    requests: list[str] = field(default_factory=list)

    def complete(self, message: str) -> str:
        self.requests.append(message)
        return f"已收到：{message}"

    def stream_response(self, message: str) -> list[str]:
        self.requests.append(message)
        return ["已", "收到：", message]


def build_client(monkeypatch, agent: object) -> ChatClient:
    """替换默认 Agent，测试 ChatClient 的入口行为。"""

    monkeypatch.setattr("app.service.client.LangChainChatAgent", lambda: agent)
    return ChatClient()


def test_complete_forwards_message(monkeypatch) -> None:
    """一次性入口应把用户消息交给默认 Agent。"""

    agent = FakeAgent()
    client = build_client(monkeypatch, agent)

    response = client.complete("第三章第二段")

    assert response == "已收到：第三章第二段"
    assert agent.requests == ["第三章第二段"]


def test_stream_response_forwards_agent_chunks(monkeypatch) -> None:
    """流式入口应保持 Agent 片段顺序并过滤空片段。"""

    agent = FakeAgent()
    client = build_client(monkeypatch, agent)

    assert list(client.stream_response("政府采购")) == [
        "已",
        "收到：",
        "政府采购",
    ]
    assert agent.requests == ["政府采购"]


class FailingAgent:
    """抛出底层异常的测试 Agent。"""

    def complete(self, message: str) -> str:
        raise RuntimeError("test failure")

    def stream_response(self, message: str) -> list[str]:
        raise RuntimeError("test failure")


def test_wraps_complete_failure(monkeypatch) -> None:
    """Agent 底层异常应转换为安全的应用异常。"""

    client = build_client(monkeypatch, FailingAgent())

    with pytest.raises(AppException) as error:
        client.complete("你好")

    assert error.value.error_code == CHAT_AGENT_FAILED


def test_wraps_stream_response_failure(monkeypatch) -> None:
    """Agent 流式异常应转换为安全的应用异常。"""

    client = build_client(monkeypatch, FailingAgent())

    with pytest.raises(AppException) as error:
        list(client.stream_response("你好"))

    assert error.value.error_code == CHAT_AGENT_FAILED
