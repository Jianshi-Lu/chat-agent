"""验证会话标题中间件：首轮生成一次、失败封顶、State 持久化。"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.service.agent.middlewares.title_generator import (
    ChatTitleState,
    ConversationTitleGenerator,
)


class CountingTitleModel:
    """记录调用次数与最后一次 prompt 的假标题模型。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_prompt: str | None = None

    def invoke(self, prompt: str) -> AIMessage:
        self.calls += 1
        self.last_prompt = prompt
        return AIMessage(content="测试标题")


class RaisingTitleModel:
    """模拟标题模型故障。"""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, prompt: str) -> AIMessage:
        self.calls += 1
        raise RuntimeError("title model unavailable")


def _build_agent(title_model: object):
    main_model = FakeMessagesListChatModel(
        responses=[
            AIMessage(content="回答一"),
            AIMessage(content="回答二"),
        ]
    )
    return create_agent(
        model=main_model,
        middleware=[ConversationTitleGenerator(model=title_model)],
        checkpointer=InMemorySaver(),
        state_schema=ChatTitleState,
    )


def test_title_generated_once_per_thread() -> None:
    """标题只在每个线程的首轮生成；后续轮次不调用标题模型。"""

    title_model = CountingTitleModel()
    agent = _build_agent(title_model)
    config = {"configurable": {"thread_id": "title-once"}}

    first = agent.invoke(
        {"messages": [HumanMessage(content="第一问")]},
        config=config,
    )
    assert first["generated_title"] == "测试标题"
    assert first.get("title_generated") is True
    assert title_model.calls == 1
    assert "第一问" in title_model.last_prompt

    state = agent.get_state(config)
    assert state.values["generated_title"] == "测试标题"
    assert state.values["title_attempted"] is True

    second = agent.invoke(
        {"messages": [HumanMessage(content="第二问")]},
        config=config,
    )
    assert title_model.calls == 1
    assert not second.get("title_generated")


def test_title_failure_marks_attempted_without_retry() -> None:
    """标题模型失败时不阻断问答，且不会在后续轮次反复重试。"""

    title_model = RaisingTitleModel()
    agent = _build_agent(title_model)
    config = {"configurable": {"thread_id": "title-fail"}}

    first = agent.invoke(
        {"messages": [HumanMessage(content="第一问")]},
        config=config,
    )
    assert first["messages"][-1].content == "回答一"
    assert first["generated_title"] is None
    assert first.get("title_generated") is False
    assert title_model.calls == 1

    state = agent.get_state(config)
    assert state.values["title_attempted"] is True
    assert state.values.get("generated_title") is None

    agent.invoke(
        {"messages": [HumanMessage(content="第二问")]},
        config=config,
    )
    assert title_model.calls == 1
