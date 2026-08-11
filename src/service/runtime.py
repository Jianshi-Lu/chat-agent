"""持有对话模型、向量服务客户端、Checkpointer 和 LangChain Agent。"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pprint import pformat
import re

import pymysql
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

from configuration import (
    CheckpointSettings,
    ContextCompressionSettings,
    VectorServiceSettings,
    get_chat_model_settings,
    get_checkpoint_settings,
    get_context_compression_settings,
    get_vector_service_settings,
)
from src.errors import (
    AppException,
    CHAT_AGENT_FAILED,
    CHAT_AGENT_UNAVAILABLE,
    CHAT_INPUT_INVALID,
)

from .agent.chat_observer import AgentObservationHandler
from .agent.middlewares.title_generator import (
    ChatTitleState,
    ConversationTitleGenerator,
)
from .agent.tools import (
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
from .vector_client import VectorServiceClient


_STATE_TRACE_ENV = "CHAT_AGENT_TRACE_STATE"
_OBSERVATION_ENV = "CHAT_AGENT_OBSERVATION_ENABLED"
# 两段式资料问答至少会经历“模型 → 记录工具 → 模型 → 资信工具 → 模型”。
# create_agent 的中间件节点同样计入 LangGraph superstep，8 会在最终回答前误杀
# 正常流程；保留与当前 LangGraph 默认值一致的有限上限，仍可阻止异常工具循环。
_AGENT_RECURSION_LIMIT = 25
_REFERENCE_TOOL_NAMES = frozenset(
    {
        "find_ingested_files",
        "read_credential_file",
        "retrieve_bid_credentials",
        "retrieve_bid_materials",
    }
)
_PUBLIC_KNOWLEDGE_TOOL_NAMES = frozenset({"retrieve", "read_knowledge_file"})
_BID_RECORD_ID_PATTERNS = (
    re.compile(r"投标记录\s*(?:ID)?\s*(?:[：:]|\s)\s*(\d+)"),
    re.compile(r"(?m)^ID\s+(\d+)(?=\s*[；;])"),
)
_EXPLICIT_BID_RECORD_ID_QUERY_PATTERN = re.compile(
    r"投标记录\s*(?:ID)?\s*(?:[：:]|\s)\s*(\d+)"
)
_PUBLIC_KNOWLEDGE_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:PUBLIC|POLICY|KNOWLEDGE)(?:[-_][A-Z0-9]+)+\b", re.IGNORECASE
)
_PUBLIC_KNOWLEDGE_NOT_FOUND_PREFIX = "未找到包含公共知识编号"
_BID_MATERIAL_INTENT_PATTERN = re.compile(
    r"投标记录|投标人|资信|报价|证书|注册资本|供应商|竞标"
)
_CREDENTIAL_MATERIAL_INTENT_PATTERN = re.compile(r"资信|文件|证书|注册资本|材料|下载")
_MAX_REFERENCED_BID_RECORDS = 12
_INTERNAL_REFERENCE_LINE_PATTERN = re.compile(
    r"(?mi)^[ \t]*(?:[-*][ \t]*)?(?:\*{1,2})?(?:文件[ \t]*)?"
    r"(?:UUID|file_uuid|document_uuid|对象存储路径|可下载路径)"
    r"(?:\*{1,2})?[：:][^\r\n]*(?:\r?\n)?"
)
_INTERNAL_REFERENCE_PARENTHESIS_PATTERN = re.compile(
    r"(?i)[（(]\s*(?:file_uuid|document_uuid|(?:资信)?文件\s*(?:UUID|ID)|"
    r"文档\s*(?:UUID|ID))\s*[：:]"
    r"[^）)\r\n]*[）)]"
)
_INTERNAL_REFERENCE_VALUE_PATTERN = re.compile(
    r"(?i)(?:file_uuid|document_uuid|(?:资信)?文件\s*(?:UUID|ID)|文档\s*(?:UUID|ID))"
    r"\s*[：:]\s*`?(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}|（内部标识已隐藏）|\d+)`?"
)
_UUID_PATTERN = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    """一次 Agent 调用的正文、标题和可验证的投标资料引用。"""

    answer: str
    title: str | None
    referenced_bid_record_ids: tuple[int, ...]


def _state_trace_enabled() -> bool:
    return os.environ.get(_STATE_TRACE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _observation_enabled() -> bool:
    return os.environ.get(_OBSERVATION_ENV, "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class Runtime:
    """持有单进程对话 Agent 的全部运行时资源。"""

    def __init__(
        self,
        vector_service_settings: VectorServiceSettings | None = None,
        context_compression_settings: ContextCompressionSettings | None = None,
        checkpoint_settings: CheckpointSettings | None = None,
    ) -> None:
        self.vector_service_settings = (
            vector_service_settings or get_vector_service_settings()
        )
        self.context_compression_settings = (
            context_compression_settings or get_context_compression_settings()
        )
        self.checkpoint_settings = checkpoint_settings or get_checkpoint_settings()
        self.vector_client: VectorServiceClient | None = None
        self.retrieval_context: RetrievalToolContext | None = None
        self.chat_model: ChatOpenAI | None = None
        self.agent: Runnable | None = None
        self.checkpointer: BaseCheckpointSaver | None = None
        self._checkpoint_connection: pymysql.Connection | None = None
        self.title_generator: ConversationTitleGenerator | None = None
        self._trace_state = _state_trace_enabled()
        self._observation_enabled = _observation_enabled()

        try:
            print("正在连接独立向量服务...", flush=True)
            self.vector_client = VectorServiceClient(self.vector_service_settings)
            self.vector_client.health()
            self.retrieval_context = RetrievalToolContext(
                vector_client=self.vector_client
            )

            print("正在初始化 LangChain Agent...", flush=True)
            chat_settings = get_chat_model_settings()
            common_model_kwargs = {
                "base_url": chat_settings.base_url,
                "api_key": chat_settings.api_key,
                "top_p": chat_settings.top_p,
                "timeout": chat_settings.timeout,
                "max_retries": chat_settings.max_retries,
                "extra_body": {
                    "top_k": chat_settings.top_k,
                    "repetition_penalty": chat_settings.repetition_penalty,
                },
            }
            self.chat_model = ChatOpenAI(
                model=chat_settings.model_name,
                temperature=chat_settings.temperature,
                max_completion_tokens=chat_settings.max_tokens,
                **common_model_kwargs,
            )
            if not self.checkpoint_settings.database_url:
                raise ValueError(
                    "环境变量 CHAT_AGENT_CHECKPOINT_DATABASE_URL 未配置，"
                    "无法初始化 MySQL Checkpointer"
                )
            print("正在初始化 MySQL Checkpointer...", flush=True)
            self._checkpoint_connection = pymysql.connect(
                **PyMySQLSaver.parse_conn_string(
                    self.checkpoint_settings.database_url
                ),
                autocommit=True,
            )
            self.checkpointer = PyMySQLSaver(self._checkpoint_connection)
            self.checkpointer.setup()
            self.title_generator = ConversationTitleGenerator(model=self.chat_model)

            summary_model = self.chat_model
            if (
                self.context_compression_settings.enabled
                and self.context_compression_settings.summary_model_name
            ):
                summary_model = ChatOpenAI(
                    model=self.context_compression_settings.summary_model_name,
                    temperature=self.context_compression_settings.summary_temperature,
                    max_completion_tokens=(
                        self.context_compression_settings.summary_max_tokens
                    ),
                    **common_model_kwargs,
                )
            middleware = [self.title_generator]
            if self.context_compression_settings.enabled:
                middleware.append(
                    SummarizationMiddleware(
                        model=summary_model,
                        trigger=(
                            "tokens",
                            self.context_compression_settings.trigger_tokens,
                        ),
                        keep=(
                            "messages",
                            self.context_compression_settings.keep_messages,
                        ),
                    )
                )
            self.agent = create_agent(
                model=self.chat_model,
                tools=[
                    retrieve,
                    list_ingested_files,
                    find_ingested_files,
                    read_knowledge_file,
                    read_credential_file,
                    list_bid_records,
                    get_bid_record,
                    retrieve_bid_materials,
                    retrieve_bid_credentials,
                ],
                middleware=middleware,
                state_schema=ChatTitleState,
                context_schema=RetrievalToolContext,
                checkpointer=self.checkpointer,
                system_prompt=chat_settings.system_prompt,
                name="local_chat_agent",
            )
        except AppException:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise AppException(
                CHAT_AGENT_UNAVAILABLE,
                module="service.runtime",
                operation="initialize",
                stage="initialize_runtime",
                message="本地知识问答运行时初始化失败",
            ) from exc

    def answer(self, question: str, *, thread_id: str) -> str:
        return self._answer(question, thread_id=thread_id)[0]

    def answer_with_title(
        self,
        question: str,
        *,
        thread_id: str,
    ) -> tuple[str, str | None]:
        return self._answer(question, thread_id=thread_id)

    def answer_with_references(
        self,
        question: str,
        *,
        thread_id: str,
    ) -> AgentAnswer:
        """返回正文及仅由实际资料工具产出的投标记录引用。"""

        return self._answer_with_references(question, thread_id=thread_id)

    def _answer(
        self,
        question: str,
        *,
        thread_id: str,
    ) -> tuple[str, str | None]:
        result = self._answer_with_references(question, thread_id=thread_id)
        return result.answer, result.title

    def _answer_with_references(
        self,
        question: str,
        *,
        thread_id: str,
    ) -> AgentAnswer:
        if not isinstance(question, str) or not question.strip():
            raise AppException(
                CHAT_INPUT_INVALID,
                module="service.runtime",
                operation="answer",
                stage="validate_input",
                message="问题不能为空",
            )
        if self.retrieval_context is None or self.agent is None:
            raise AppException(
                CHAT_AGENT_UNAVAILABLE,
                module="service.runtime",
                operation="answer",
                stage="validate_runtime",
                message="本地知识问答运行时尚未初始化",
            )

        config = self._thread_config(thread_id)
        observer: AgentObservationHandler | None = None
        if getattr(self, "_observation_enabled", True):
            observer = AgentObservationHandler(
                thread_id=thread_id,
                question=question,
            )
            config["callbacks"] = [observer]

        try:
            identifiers = {
                match.group(0).upper()
                for match in _PUBLIC_KNOWLEDGE_IDENTIFIER_PATTERN.finditer(question)
            }
            current_public_evidence: str | None = None
            current_bid_evidence: list[str] = []
            current_referenced_bid_record_ids: tuple[int, ...] | None = None
            if identifiers:
                # Checkpointer 会向模型恢复整个对话；仅靠提示词时，模型可能把上一轮
                # 的答案当作当前事实而跳过工具调用。稳定公共知识编号需要在每轮从
                # vector-service 重新取证，并以本轮证据覆盖历史上下文。
                current_public_evidence = self.retrieval_context.vector_client.search_knowledge(
                    [question]
                )
                is_public_knowledge_only = not bool(
                    _BID_MATERIAL_INTENT_PATTERN.search(question)
                )
                if (
                    is_public_knowledge_only
                    and current_public_evidence.startswith(
                        _PUBLIC_KNOWLEDGE_NOT_FOUND_PREFIX
                    )
                ):
                    return AgentAnswer(
                        answer=self._public_knowledge_not_found_answer(identifiers),
                        title=None,
                        referenced_bid_record_ids=(),
                    )

            explicit_bid_record_ids = tuple(
                dict.fromkeys(
                    int(match.group(1))
                    for match in _EXPLICIT_BID_RECORD_ID_QUERY_PATTERN.finditer(question)
                )
            )
            if explicit_bid_record_ids:
                # 与公共知识编号一样，明确的投标记录查询也不能复用 checkpoint
                # 中已经过期的回答。每轮重新读取记录；涉及资信材料时再限定到
                # 当前记录检索资料，并由这份新证据决定是否展示下载卡片。
                credential_query = bool(
                    _CREDENTIAL_MATERIAL_INTENT_PATTERN.search(question)
                )
                current_references: list[int] = []
                missing_bid_record_ids: list[int] = []
                for bid_record_id in explicit_bid_record_ids:
                    bid_record_evidence = (
                        self.retrieval_context.vector_client.get_bid_record(
                            bid_record_id
                        )
                    )
                    current_bid_evidence.append(bid_record_evidence)
                    if self._is_missing_bid_record_evidence(
                        bid_record_evidence,
                        bid_record_id,
                    ):
                        missing_bid_record_ids.append(bid_record_id)
                        continue
                    if not credential_query:
                        continue
                    credential_evidence = (
                        self.retrieval_context.vector_client.search_bid_credentials(
                            bid_record_id,
                            [question],
                        )
                    )
                    current_bid_evidence.append(credential_evidence)
                    if bid_record_id in self._bid_record_ids_from_text(
                        credential_evidence
                    ):
                        current_references.append(bid_record_id)
                if credential_query:
                    current_referenced_bid_record_ids = tuple(current_references)
                if len(missing_bid_record_ids) == len(explicit_bid_record_ids) and not identifiers:
                    return AgentAnswer(
                        answer=self._bid_record_not_found_answer(
                            missing_bid_record_ids
                        ),
                        title=None,
                        referenced_bid_record_ids=(),
                    )

            messages: list[HumanMessage | SystemMessage] = [
                HumanMessage(content=question)
            ]
            current_evidence = []
            if current_public_evidence is not None:
                current_evidence.append(current_public_evidence)
            current_evidence.extend(current_bid_evidence)
            if current_evidence:
                current_evidence_text = "\n\n".join(current_evidence)
                messages.insert(
                    0,
                    SystemMessage(
                        content=(
                            "本轮知识和投标资料已由检索服务重新取证。以下内容优先于"
                            "任何历史对话；只能据此回答，不得用历史回答替代，也不"
                            "得编造未出现的事实。\n\n"
                            f"{current_evidence_text}"
                        )
                    ),
                )
            result = self.agent.invoke(
                {"messages": messages},
                context=self.retrieval_context,
                config=config,
            )
            if getattr(self, "_trace_state", False):
                self._print_state(f"result, thread_id={thread_id}", result)
                get_state = getattr(self.agent, "get_state", None)
                if callable(get_state):
                    self._print_state(
                        f"checkpoint, thread_id={thread_id}",
                        get_state(config),
                    )
            answer = self._ground_public_knowledge_answer(
                self._sanitize_user_answer(self._result_text(result)),
                result,
                question,
                current_public_evidence=current_public_evidence,
            )
            referenced_bid_record_ids = self._referenced_bid_record_ids(
                result,
                question=question,
                current_referenced_bid_record_ids=current_referenced_bid_record_ids,
            )
            title: str | None = None
            if isinstance(result, dict) and result.get("title_generated"):
                generated = result.get("generated_title")
                if isinstance(generated, str) and generated.strip():
                    title = generated.strip()
        except AppException:
            raise
        except Exception as exc:
            log.exception(
                "Agent 调用失败: threadId=%s, exceptionType=%s",
                thread_id,
                type(exc).__name__,
            )
            raise AppException(
                CHAT_AGENT_FAILED,
                module="service.runtime",
                operation="answer",
                stage="invoke_agent",
                message="本地 LangChain Agent 调用失败",
            ) from exc
        finally:
            if observer is not None:
                observer.summary()

        if not isinstance(answer, str) or not answer.strip():
            raise AppException(
                CHAT_AGENT_FAILED,
                module="service.runtime",
                operation="answer",
                stage="validate_response",
                message="本地 Agent 返回了空回答",
            )
        return AgentAnswer(
            answer=answer,
            title=title,
            referenced_bid_record_ids=referenced_bid_record_ids,
        )

    @staticmethod
    def _referenced_bid_record_ids(
        result: object,
        *,
        question: str,
        current_referenced_bid_record_ids: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """从本轮真正执行的资料工具输出中提取投标记录 ID。

        这不是从模型自然语言回答中猜测 ID：只有与资信资料有关的 ToolMessage
        才会参与，因此前端的下载入口始终对应一次实际检索到的资料范围。
        """

        if current_referenced_bid_record_ids is not None:
            return current_referenced_bid_record_ids
        if not isinstance(result, dict):
            return ()
        messages = Runtime._current_turn_messages(result, question=question)
        if not messages:
            return ()

        # 文件卡片只能服务于用户正在询问的投标资料。公共知识编号（例如
        # ``PUBLIC-E2E-8047``）中的尾号不是 bid_record_id；即使模型错误调用了
        # 投标资料工具，也不能把无关资信文件呈现在该公共知识回答下。
        is_public_knowledge_only = bool(
            _PUBLIC_KNOWLEDGE_IDENTIFIER_PATTERN.search(question)
        ) and not bool(_BID_MATERIAL_INTENT_PATTERN.search(question))
        if is_public_knowledge_only:
            return ()

        explicit_ids = {
            int(match.group(1))
            for match in _EXPLICIT_BID_RECORD_ID_QUERY_PATTERN.finditer(question)
        }
        found: list[int] = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            if getattr(message, "name", None) not in _REFERENCE_TOOL_NAMES:
                continue
            content = Runtime._message_text(message)
            for bid_record_id in Runtime._bid_record_ids_from_text(content):
                if (
                    bid_record_id in found
                    or (explicit_ids and bid_record_id not in explicit_ids)
                ):
                    continue
                found.append(bid_record_id)
                if len(found) >= _MAX_REFERENCED_BID_RECORDS:
                    return tuple(found)
        return tuple(found)

    @staticmethod
    def _bid_record_ids_from_text(content: str) -> tuple[int, ...]:
        found: list[int] = []
        for pattern in _BID_RECORD_ID_PATTERNS:
            for match in pattern.finditer(content):
                bid_record_id = int(match.group(1))
                if bid_record_id > 0 and bid_record_id not in found:
                    found.append(bid_record_id)
        return tuple(found)

    @staticmethod
    def _is_missing_bid_record_evidence(content: str, bid_record_id: int) -> bool:
        return (
            f"未找到投标记录 {bid_record_id}" in content
            or f"未找到投标记录 ID：{bid_record_id}" in content
        )

    @staticmethod
    def _bid_record_not_found_answer(bid_record_ids: list[int]) -> str:
        labels = "、".join(f"{bid_record_id}" for bid_record_id in bid_record_ids)
        return (
            f"当前投标资料中未找到投标记录 ID：{labels}，"
            "无法依据已删除或不存在的记录提供投标人、报价或资信材料。"
        )

    @staticmethod
    def _current_turn_messages(result: object, *, question: str) -> list[object]:
        """返回当前用户问题之后产生的消息，避免历史工具结果充当本轮证据。

        LangGraph 的 checkpointer 会在 ``invoke`` 返回值中带回整个会话状态。
        因此同一 ``thread_id`` 里已经删除的文件、或上一轮命中的投标资料，不能
        因为仍存在于历史 ``ToolMessage`` 中而继续生成下载卡片或通过事实校验。
        """

        if not isinstance(result, dict):
            return []
        messages = result.get("messages")
        if not isinstance(messages, list):
            return []

        normalized_question = question.strip()
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, HumanMessage):
                continue
            if Runtime._message_text(message).strip() == normalized_question:
                return messages[index + 1 :]

        # 单元测试和无持久化运行模式可能只提供本轮消息；此时列表本身就是本轮。
        return messages

    @staticmethod
    def _ground_public_knowledge_answer(
        answer: str,
        result: object,
        question: str,
        *,
        current_public_evidence: str | None = None,
    ) -> str:
        """拒绝对未在实际公共知识工具输出中出现的稳定编号进行臆答。"""

        identifiers = {
            match.group(0).upper()
            for match in _PUBLIC_KNOWLEDGE_IDENTIFIER_PATTERN.finditer(question)
        }
        if not identifiers or not isinstance(result, dict):
            return answer
        messages = Runtime._current_turn_messages(result, question=question)
        if not messages:
            return answer
        evidence_parts = [] if current_public_evidence is None else [current_public_evidence]
        evidence_parts.extend(
            Runtime._message_text(message)
            for message in messages
            if isinstance(message, ToolMessage)
            and getattr(message, "name", None) in _PUBLIC_KNOWLEDGE_TOOL_NAMES
        )
        evidence = "\n".join(evidence_parts)
        if any(identifier.casefold() in evidence.casefold() for identifier in identifiers):
            return answer
        return Runtime._public_knowledge_not_found_answer(identifiers)

    @staticmethod
    def _public_knowledge_not_found_answer(identifiers: set[str]) -> str:
        quoted_identifiers = "、".join(
            f"“{identifier}”" for identifier in sorted(identifiers)
        )
        return (
            f"当前已入库公共知识中未检索到编号{quoted_identifiers}，"
            "无法依据现有资料给出该编号的具体答案。"
        )

    @staticmethod
    def _print_state(label: str, state: object) -> None:
        print(
            f"\n===== Agent State ({label}) =====\n"
            f"{pformat(state, width=120, sort_dicts=False)}\n"
            "===== End Agent State =====",
            flush=True,
        )

    @staticmethod
    def _result_text(result: object) -> str:
        if not isinstance(result, dict):
            raise TypeError("Agent 返回结果必须是包含 messages 的字典")
        messages = result.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Agent 返回结果中缺少有效 messages")
        final_message = messages[-1]
        if not isinstance(final_message, (AIMessage, AIMessageChunk)):
            raise TypeError("Agent 最后一条消息不是模型回答")
        text = Runtime._message_text(final_message)
        if not text:
            raise ValueError("Agent 最终回答不包含可用文本")
        return text

    @staticmethod
    def _sanitize_user_answer(answer: str) -> str:
        """移除模型可能复述的内部定位信息，下载只能走前端受控入口。"""

        sanitized = _INTERNAL_REFERENCE_LINE_PATTERN.sub("", answer)
        sanitized = _INTERNAL_REFERENCE_PARENTHESIS_PATTERN.sub("", sanitized)
        sanitized = _INTERNAL_REFERENCE_VALUE_PATTERN.sub("", sanitized)
        sanitized = _UUID_PATTERN.sub("（内部标识已隐藏）", sanitized)
        sanitized = re.sub(r"[（(]\s*[）)]", "", sanitized)
        sanitized = re.sub(r"[，,]\s*([。；;])", r"\1", sanitized)
        return re.sub(r"\n{3,}", "\n\n", sanitized).strip()

    @staticmethod
    def _message_text(message: object) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    texts.append(block["text"])
            return "".join(texts)
        return ""

    @staticmethod
    def _thread_config(
        thread_id: str,
        callbacks: list | None = None,
    ) -> dict[str, object]:
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id 不能为空")
        config: dict[str, object] = {
            "configurable": {"thread_id": thread_id.strip()},
            "recursion_limit": _AGENT_RECURSION_LIMIT,
        }
        if callbacks:
            config["callbacks"] = callbacks
        return config

    def close(self) -> None:
        checkpoint_connection = self._checkpoint_connection
        vector_client = self.vector_client
        self._checkpoint_connection = None
        self.checkpointer = None
        self.title_generator = None
        self.vector_client = None
        self.retrieval_context = None
        self.agent = None
        self.chat_model = None

        if checkpoint_connection is not None:
            checkpoint_connection.close()
        if vector_client is not None:
            vector_client.close()


__all__ = ["AgentAnswer", "Runtime"]
