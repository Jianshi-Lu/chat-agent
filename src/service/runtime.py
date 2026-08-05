"""管理本地知识问答服务运行期间的模型、向量库和 LangChain Agent。

``Runtime`` 在服务启动时一次性创建 Embedding 模型、持久化 Chroma、检索器、
ChatOpenAI 客户端、checkpointer 和 LangChain Agent；请求处理只复用这些对象。
上下文压缩由 Agent 的官方 Middleware 执行，服务关闭时由 ``close`` 统一释放
Chroma、Agent、模型引用和设备缓存。
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pprint import pformat
from threading import Lock

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from configuration import (
    ContextCompressionSettings,
    IngestionSettings,
    RetrievalSettings,
    get_chat_model_settings,
    get_context_compression_settings,
    get_ingestion_settings,
    get_retrieval_settings,
)

from src.errors import (
    AppException,
    CHAT_AGENT_FAILED,
    CHAT_AGENT_UNAVAILABLE,
    CHAT_INPUT_INVALID,
)

from .agent.tools import RetrievalToolContext, exact_paragraph_search, retrieve
from .ingestion import IngestionPipeline, KnowledgeDocument


_STATE_TRACE_ENV = "CHAT_AGENT_TRACE_STATE"


def _state_trace_enabled() -> bool:
    """读取是否在终端输出本次 Agent 调用的完整 State。"""

    return os.environ.get(_STATE_TRACE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class Runtime:
    """持有单进程知识问答服务的全部运行时资源。"""

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        retrieval_settings: RetrievalSettings | None = None,
        context_compression_settings: ContextCompressionSettings | None = None,
    ) -> None:
        """加载 Embedding、持久化 Chroma、政策数据和带压缩能力的 Agent。"""

        self.settings = settings or get_ingestion_settings()
        self.retrieval_settings = retrieval_settings or get_retrieval_settings()
        self.context_compression_settings = (
            context_compression_settings or get_context_compression_settings()
        )
        self.ingestion: IngestionPipeline | None = None
        self.retrieval_context: RetrievalToolContext | None = None
        self.chat_model: ChatOpenAI | None = None
        self.agent: Runnable | None = None
        self.checkpointer: BaseCheckpointSaver | None = None
        self.chunks: list[Document] = []
        self.chunk_count = 0
        self._ingestion_lock = Lock()
        self._trace_state = _state_trace_enabled()

        try:
            print("正在加载本地 Embedding 模型和持久化 Chroma...", flush=True)
            self.ingestion = IngestionPipeline(
                model_path=self.settings.embedding_model_path,
                chroma_path=self.settings.chroma_path,
                collection_name=self.settings.collection_name,
                embedding_batch_size=self.settings.embedding_batch_size,
                query_embedding_batch_size=self.settings.query_embedding_batch_size,
            )
            existing_chunk_count = len(
                self.ingestion.vector_store.get(include=[]).get("ids", [])
            )
            if existing_chunk_count:
                self.chunk_count = existing_chunk_count
                print(
                    f"检测到已有 Chroma 数据：{existing_chunk_count} 个文本块，跳过向量化。",
                    flush=True,
                )
            else:
                print(
                    f"正在读取 {self.settings.source_path} 并生成文档向量...",
                    flush=True,
                )
                self.chunks = self.ingestion.run(
                    self.settings.source_path,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
                self.chunk_count = len(self.chunks)

            self.retrieval_context = RetrievalToolContext(
                vector_store=self.ingestion.vector_store,
                embeddings=self.ingestion.embeddings,
                top_k=self.retrieval_settings.top_k,
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
            if self.context_compression_settings.enabled:
                self.checkpointer = InMemorySaver()
            middleware: list[object] = []
            if self.context_compression_settings.enabled:
                summary_model = ChatOpenAI(
                    model=(
                        self.context_compression_settings.summary_model_name
                        or chat_settings.model_name
                    ),
                    temperature=self.context_compression_settings.summary_temperature,
                    max_completion_tokens=(
                        self.context_compression_settings.summary_max_tokens
                    ),
                    **common_model_kwargs,
                )
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
                tools=[exact_paragraph_search, retrieve],
                middleware=middleware,
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
        """按 thread_id 处理问题，并由 checkpointer 管理短期对话历史。"""

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

        try:
            config = (
                self._thread_config(thread_id)
                if self.checkpointer is not None
                else None
            )
            invoke_kwargs: dict[str, object] = {
                "context": self.retrieval_context,
            }
            if config is not None:
                invoke_kwargs["config"] = config
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                **invoke_kwargs,
            )
            # 单测或外部注入的轻量 Runtime 可能绕过 __init__，此时默认关闭调试输出。
            if getattr(self, "_trace_state", False):
                self._print_state(f"result, thread_id={thread_id}", result)
                if config is not None:
                    get_state = getattr(self.agent, "get_state", None)
                    if callable(get_state):
                        self._print_state(
                            f"checkpoint, thread_id={thread_id}",
                            get_state(config),
                        )
            answer = self._result_text(result)
        except AppException:
            raise
        except Exception as exc:
            raise AppException(
                CHAT_AGENT_FAILED,
                module="service.runtime",
                operation="answer",
                stage="invoke_agent",
                message="本地 LangChain Agent 调用失败",
            ) from exc

        if not isinstance(answer, str) or not answer.strip():
            raise AppException(
                CHAT_AGENT_FAILED,
                module="service.runtime",
                operation="answer",
                stage="validate_response",
                message="本地 Agent 返回了空回答",
            )
        return answer

    @staticmethod
    def _print_state(label: str, state: object) -> None:
        """以完整可读形式输出 Agent State；仅由显式 trace 开关触发。"""

        print(
            f"\n===== Agent State ({label}) =====\n"
            f"{pformat(state, width=120, sort_dicts=False)}\n"
            "===== End Agent State =====",
            flush=True,
        )

    def vectorize_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> list[tuple[str, int]]:
        """在一次写锁内向量化多个知识文件，并在失败时按 UUID 补偿。"""

        documents = list(documents)
        if not documents:
            raise AppException(
                CHAT_INPUT_INVALID,
                module="service.runtime",
                operation="vectorize_documents",
                stage="validate_input",
                message="待向量化知识文件不能为空",
            )
        if self.ingestion is None:
            raise AppException(
                CHAT_AGENT_UNAVAILABLE,
                module="service.runtime",
                operation="vectorize_documents",
                stage="validate_runtime",
                message="知识向量化 Runtime 尚未初始化",
            )

        with self._ingestion_lock:
            results: list[tuple[str, int]] = []
            try:
                for document in documents:
                    chunks = self.ingestion.vectorize_document(
                        document,
                        chunk_size=self.settings.chunk_size,
                        chunk_overlap=self.settings.chunk_overlap,
                    )
                    results.append((document.document_uuid, len(chunks)))
            except Exception:
                for document in documents:
                    try:
                        self.ingestion.delete_document(document.document_uuid)
                    except AppException:
                        pass
                self._refresh_chunk_count()
                raise

            self._refresh_chunk_count()
            return results

    def delete_document(self, document_uuid: str) -> int:
        """在写锁内幂等删除指定 document_uuid 的向量。"""

        if self.ingestion is None:
            raise AppException(
                CHAT_AGENT_UNAVAILABLE,
                module="service.runtime",
                operation="delete_document",
                stage="validate_runtime",
                message="知识向量化 Runtime 尚未初始化",
            )
        with self._ingestion_lock:
            deleted_count = self.ingestion.delete_document(document_uuid)
            self._refresh_chunk_count()
            return deleted_count

    def _refresh_chunk_count(self) -> None:
        """从当前 Chroma collection 刷新运行时 chunk 数量。"""

        if self.ingestion is None:
            self.chunk_count = 0
            return
        self.chunk_count = len(
            self.ingestion.vector_store.get(include=[]).get("ids", [])
        )

    @staticmethod
    def _result_text(result: object) -> str:
        """从官方 Agent 状态结果中提取最后一条模型消息文本。"""

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
    def _message_text(message: object) -> str:
        """提取 LangChain 消息中的文本内容。"""

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
    def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
        """构造 LangGraph checkpointer 使用的线程配置。"""

        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id 不能为空")
        return {"configurable": {"thread_id": thread_id.strip()}}

    def close(self) -> None:
        """释放 Agent、模型、Chroma 客户端和设备缓存。"""

        ingestion = self.ingestion
        self.ingestion = None
        self.retrieval_context = None
        self.agent = None
        self.checkpointer = None
        self.chat_model = None
        self.chunks = []
        self.chunk_count = 0

        if ingestion is not None:
            ingestion.close()

__all__ = ["Runtime"]
