"""LangChain Agent 调用观测回调。

每次对话请求创建一个 ``AgentObservationHandler`` 实例，通过 LangChain 官方
callback 机制收集模型调用、工具/检索调用、token 与耗时信息，并以项目统一的
``log.info("...: key=%s, ...")`` 格式输出汇总日志。问题、prompt 与回答只记录
截断摘要，避免敏感内容进入日志。
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

log = logging.getLogger(__name__)

_MAX_LOG_TEXT = 200


def _truncate(text: object, limit: int = _MAX_LOG_TEXT) -> str:
    """把任意文本压成单行截断摘要。"""

    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


class AgentObservationHandler(BaseCallbackHandler):
    """收集一次 Agent 调用的运行观测数据并输出汇总日志。"""

    name = "chat_agent_observation"

    def __init__(
        self,
        *,
        thread_id: str,
        question: str | None = None,
    ) -> None:
        self.thread_id = thread_id
        self.request_id = uuid.uuid4().hex[:8]
        self.question = question
        self._started_at = time.perf_counter()
        self._llm_starts: dict[uuid.UUID, float] = {}
        self._tool_starts: dict[uuid.UUID, float] = {}
        self.llm_calls = 0
        self.total_tokens = 0
        self.tool_calls = 0
        self.retrieval_calls = 0
        self.agent_steps = 0
        self.error_count = 0

    # ---------- 模型调用 ----------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self.llm_calls += 1
        self._llm_starts[run_id] = time.perf_counter()
        model = serialized.get("name") or kwargs.get("model") or ""
        last_text = ""
        if messages and messages[-1]:
            last_text = _truncate(getattr(messages[-1][-1], "content", ""))
        log.debug(
            "模型调用开始: threadId=%s, requestId=%s, model=%s, input=%s",
            self.thread_id,
            self.request_id,
            model,
            last_text,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        started_at = self._llm_starts.pop(run_id, None)
        tokens = self._extract_tokens(response)
        if tokens:
            self.total_tokens += tokens
        elapsed_ms = (
            round((time.perf_counter() - started_at) * 1000)
            if started_at is not None
            else 0
        )
        log.debug(
            "模型调用结束: threadId=%s, requestId=%s, tokens=%s, elapsedMs=%s",
            self.thread_id,
            self.request_id,
            tokens,
            elapsed_ms,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_starts.pop(run_id, None)
        self.error_count += 1
        log.warning(
            "模型调用失败: threadId=%s, requestId=%s, reason=%s",
            self.thread_id,
            self.request_id,
            error,
        )

    # ---------- 工具 / 检索 ----------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self.tool_calls += 1
        self._tool_starts[run_id] = time.perf_counter()
        tool = serialized.get("name") or ""
        log.debug(
            "工具调用开始: threadId=%s, requestId=%s, tool=%s, input=%s",
            self.thread_id,
            self.request_id,
            tool,
            _truncate(input_str),
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        started_at = self._tool_starts.pop(run_id, None)
        elapsed_ms = (
            round((time.perf_counter() - started_at) * 1000)
            if started_at is not None
            else 0
        )
        log.debug(
            "工具调用结束: threadId=%s, requestId=%s, outputLength=%s, "
            "elapsedMs=%s",
            self.thread_id,
            self.request_id,
            len(str(output)),
            elapsed_ms,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_starts.pop(run_id, None)
        self.error_count += 1
        log.warning(
            "工具调用失败: threadId=%s, requestId=%s, reason=%s",
            self.thread_id,
            self.request_id,
            error,
        )

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        **kwargs: Any,
    ) -> None:
        self.retrieval_calls += 1
        log.debug(
            "检索开始: threadId=%s, requestId=%s, query=%s",
            self.thread_id,
            self.request_id,
            _truncate(query),
        )

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        **kwargs: Any,
    ) -> None:
        log.debug(
            "检索结束: threadId=%s, requestId=%s, documentCount=%s",
            self.thread_id,
            self.request_id,
            len(documents),
        )

    # ---------- Agent 决策 ----------

    def on_agent_action(
        self,
        action: Any,
        **kwargs: Any,
    ) -> None:
        self.agent_steps += 1
        tool = getattr(action, "tool", "") or ""
        log.debug(
            "Agent 决策: threadId=%s, requestId=%s, tool=%s, step=%s",
            self.thread_id,
            self.request_id,
            tool,
            self.agent_steps,
        )

    def summary(self) -> None:
        """输出本次请求的观测汇总日志。"""

        elapsed_ms = round((time.perf_counter() - self._started_at) * 1000)
        log.info(
            "对话请求明细: threadId=%s, requestId=%s, modelCalls=%s, "
            "totalTokens=%s, toolCalls=%s, retrievalCalls=%s, "
            "agentSteps=%s, errorCount=%s, elapsedMs=%s",
            self.thread_id,
            self.request_id,
            self.llm_calls,
            self.total_tokens,
            self.tool_calls,
            self.retrieval_calls,
            self.agent_steps,
            self.error_count,
            elapsed_ms,
        )

    @staticmethod
    def _extract_tokens(response: LLMResult) -> int:
        """从 LLMResult 中尽量读取 token 用量。"""

        tokens = 0
        for group in getattr(response, "generations", []):
            for generation in group:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if not isinstance(usage, dict):
                    continue
                total = usage.get("total_tokens")
                if total is None:
                    total = (usage.get("input_tokens") or 0) + (
                        usage.get("output_tokens") or 0
                    )
                tokens += int(total or 0)
        if tokens:
            return tokens

        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") or {}
        return int(token_usage.get("total_tokens") or 0)
