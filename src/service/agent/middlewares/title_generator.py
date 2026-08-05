"""会话标题自动生成中间件。

基于 LangChain 官方 ``after_agent`` 钩子和自定义 State Schema：每个线程只在
第一轮 Agent 执行结束后生成一次标题。``title_attempted`` 随 Checkpointer
持久化，后续轮次直接短路；``title_generated`` 是 Ephemeral 通道，只在生成
当轮出现在最终 State 中，用于告诉 Runtime 是否要把新标题返回给 Java。
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.channels.ephemeral_value import EphemeralValue
from typing_extensions import Annotated, NotRequired


_MAX_TITLE_LENGTH = 200

_TITLE_PROMPT = (
    "根据用户的第一句话，生成一个简短的中文会话标题，不超过30个字；"
    "直接输出标题本身，不要引号、标点、换行或解释。\n\n用户：{question}"
)


class ChatTitleState(AgentState):
    """带标题生成通道的 Agent State。"""

    generated_title: NotRequired[str | None]
    title_attempted: NotRequired[bool]
    title_generated: NotRequired[Annotated[bool, EphemeralValue]]


class ConversationTitleGenerator(AgentMiddleware):
    """Agent 首轮结束后自动生成会话标题。"""

    state_schema = ChatTitleState

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def after_agent(
        self,
        state: ChatTitleState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """首轮结束后生成标题；失败也标记已尝试，避免每轮重试。"""

        if state.get("title_attempted"):
            return None

        title = self._generate_title(state)
        return {
            "generated_title": title,
            "title_attempted": True,
            "title_generated": title is not None,
        }

    def _generate_title(self, state: ChatTitleState) -> str | None:
        """用第一条用户问题调用模型生成标题；任何失败都降级为 None。"""

        question = self._first_user_question(state)
        if not question:
            return None
        try:
            response = self.model.invoke(_TITLE_PROMPT.format(question=question))
            content = getattr(response, "content", "")
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "")
                    if isinstance(block, dict)
                    else str(block)
                    for block in content
                )
            title = str(content).strip().strip('"\'“”‘’').strip()
            return title[:_MAX_TITLE_LENGTH] or None
        except Exception:
            return None

    @staticmethod
    def _first_user_question(state: ChatTitleState) -> str | None:
        """提取 State 中第一条非空 HumanMessage 文本。"""

        for message in state.get("messages", []):
            if not isinstance(message, HumanMessage):
                continue
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
        return None
