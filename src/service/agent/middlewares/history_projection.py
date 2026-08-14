"""在不改写原始 Checkpoint 的前提下投影模型可见的历史消息。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    SummarizationMiddleware,
)
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing_extensions import override


def project_messages_for_model(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """返回本轮模型可见的消息视图，不修改传入的 Checkpoint State。

    同一 ``thread_id`` 在当前服务中串行调用，因此最后一条普通
    ``HumanMessage`` 是当前调用的起点。此前每个用户轮次只有在以不含
    ``tool_calls`` 的 ``AIMessage`` 结束时才被视为完成；没有最终回答的
    旧轮次只从本次模型输入隐藏，仍保留在 Checkpoint 历史中供恢复使用。
    """

    message_list = list(messages)
    current_start = _current_turn_start(message_list)
    if current_start is None:
        return message_list

    history = message_list[:current_start]
    current_turn = message_list[current_start:]
    return [*_completed_history(history), *current_turn]


class CheckpointHistoryProjectionMiddleware(AgentMiddleware):
    """仅替换单次模型请求的消息视图，不返回任何 State 清理更新。"""

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(
            request.override(messages=project_messages_for_model(request.messages))
        )


class ProjectedSummarizationMiddleware(SummarizationMiddleware):
    """让既有摘要器也只读取完成的历史，避免悬挂任务进入摘要正文。

    ``SummarizationMiddleware`` 本身会按既有机制把摘要写入新的 Checkpoint
    快照；本类不额外清理旧状态。需要恢复悬挂任务时，仍可从 Checkpointer 的
    历史快照取得压缩前的原始消息。
    """

    @override
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        messages = project_messages_for_model(state["messages"])
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(
            messages,
            cutoff_index,
        )
        summary = self._create_summary(messages_to_summarize)
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *self._build_new_messages(summary),
                *preserved_messages,
            ]
        }

    @override
    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        messages = project_messages_for_model(state["messages"])
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(
            messages,
            cutoff_index,
        )
        summary = await self._acreate_summary(messages_to_summarize)
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *self._build_new_messages(summary),
                *preserved_messages,
            ]
        }


def _current_turn_start(messages: Sequence[AnyMessage]) -> int | None:
    """定位当前用户轮次，并把紧邻的动态证据 SystemMessage 一起保留。"""

    for index in range(len(messages) - 1, -1, -1):
        if not _is_user_turn_start(messages[index]):
            continue
        start = index
        while start > 0 and isinstance(messages[start - 1], SystemMessage):
            start -= 1
        return start
    return None


def _completed_history(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """从历史中保留已真正结束的用户轮次和非轮次上下文消息。"""

    projected: list[AnyMessage] = []
    index = 0
    while index < len(messages):
        turn_start = _next_turn_start(messages, index)
        if turn_start is None:
            projected.extend(messages[index:])
            break

        projected.extend(messages[index:turn_start])
        next_turn_start = _next_turn_start(messages, turn_start + 1)
        turn_end = len(messages) if next_turn_start is None else next_turn_start
        turn = messages[turn_start:turn_end]
        if _turn_is_complete(turn):
            projected.extend(turn)
        index = turn_end
    return projected


def _next_turn_start(messages: Sequence[AnyMessage], start: int) -> int | None:
    """找到从 ``start`` 开始的下一用户轮次（含其前置动态证据）。"""

    for index in range(start, len(messages)):
        if not _is_user_turn_start(messages[index]):
            continue
        turn_start = index
        while turn_start > start and isinstance(messages[turn_start - 1], SystemMessage):
            turn_start -= 1
        return turn_start
    return None


def _is_user_turn_start(message: AnyMessage) -> bool:
    """摘要消息是上下文，不应被当成待完成的用户业务轮次。"""

    return isinstance(message, HumanMessage) and (
        message.additional_kwargs.get("lc_source") != "summarization"
    )


def _turn_is_complete(turn: Sequence[AnyMessage]) -> bool:
    """最后一个有效 Agent 消息无 tool_calls 时，该轮在 Agent 层已结束。"""

    for message in reversed(turn):
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, AIMessage):
            return not message.tool_calls
        return False
    return False
