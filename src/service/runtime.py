"""管理本地知识问答服务运行期间的模型、向量库和 LangChain Agent。

``Runtime`` 在服务启动时一次性创建 Embedding 模型、持久化 Chroma、检索器、
ChatOpenAI 客户端、checkpointer 和 LangChain Agent；请求处理只复用这些对象。
上下文压缩由 Agent 的官方 Middleware 执行，服务关闭时由 ``close`` 统一释放
Chroma、Agent、模型引用和设备缓存。
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from configuration import (
    ContextCompressionSettings,
    KnowledgeRuntimeSettings,
    RetrievalSettings,
    get_chat_model_settings,
    get_context_compression_settings,
    get_knowledge_runtime_settings,
    get_retrieval_settings,
)

from src.errors import (
    AppException,
    CHAT_AGENT_FAILED,
    CHAT_AGENT_UNAVAILABLE,
    CHAT_INPUT_INVALID,
)

from .agent.tools import build_retriever_tool, exact_paragraph_search
from .ingestion import IngestionService
from .retrieval import Retriever


_DEFAULT_SYSTEM_PROMPT = (
    "你是政策资料问答助手。涉及政策事实、政策定义、政策要求或政策来源的问题，"
    "必须优先调用 retrieve 获取知识库片段，再依据检索结果回答；"
    "如果上下文不足以支持结论，必须明确说明知识库中没有足够信息，"
    "不要凭空补充事实。已注册的工具可能包含占位实现，不能把占位结果"
    "当作真实政策事实。回答尽量简洁，并在适合时指出资料来源。"
)


class Runtime:
    """持有单进程知识问答服务的全部运行时资源。"""

    def __init__(
        self,
        settings: KnowledgeRuntimeSettings | None = None,
        retrieval_settings: RetrievalSettings | None = None,
        context_compression_settings: ContextCompressionSettings | None = None,
    ) -> None:
        """加载 Embedding、持久化 Chroma、政策数据和带压缩能力的 Agent。"""

        self.settings = settings or get_knowledge_runtime_settings()
        self.retrieval_settings = retrieval_settings or get_retrieval_settings()
        self.context_compression_settings = (
            context_compression_settings or get_context_compression_settings()
        )
        self.ingestion: IngestionService | None = None
        self.retriever: Retriever | None = None
        self.retriever_tool: BaseTool | None = None
        self.chat_model: ChatOpenAI | None = None
        self.agent: Runnable | None = None
        self.checkpointer: BaseCheckpointSaver | None = None
        self.chunks: list[Any] = []
        self.chunk_count = 0

        try:
            print("正在加载本地 Embedding 模型和持久化 Chroma...", flush=True)
            self.ingestion = IngestionService(
                model_path=self.settings.embedding_model_path,
                chroma_path=self.settings.chroma_path,
            )
            self.retriever = Retriever(self.ingestion)

            existing_chunk_count = self.ingestion.policy_chunk_collection.count()
            if existing_chunk_count:
                self.chunk_count = existing_chunk_count
                print(
                    f"检测到已有 Chroma 数据：{existing_chunk_count} 个文本块，跳过向量化。",
                    flush=True,
                )
            else:
                print("正在读取 policies.json 并生成文档向量...", flush=True)
                self.chunks = self.ingestion.ingest(self.settings.policies_path)
                self.chunk_count = len(self.chunks)

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
            self.retriever_tool = build_retriever_tool(
                self.retriever,
                top_k=self.retrieval_settings.top_k,
            )
            middleware: list[object] = []
            if self.context_compression_settings.enabled:
                summary_model = ChatOpenAI(
                    model=(
                        self.context_compression_settings.summary_model_name
                        or chat_settings.model_name
                    ),
                    temperature=0.0,
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
                tools=[exact_paragraph_search, self.retriever_tool],
                middleware=middleware,
                checkpointer=self.checkpointer,
                system_prompt=chat_settings.system_prompt or _DEFAULT_SYSTEM_PROMPT,
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

    def answer(self, question: str, *, thread_id: str = "default") -> str:
        """按 thread_id 处理问题，并由 checkpointer 管理短期对话历史。"""

        if not isinstance(question, str) or not question.strip():
            raise AppException(
                CHAT_INPUT_INVALID,
                module="service.runtime",
                operation="answer",
                stage="validate_input",
                message="问题不能为空",
            )
        if self.retriever is None or self.retriever_tool is None or self.agent is None:
            raise AppException(
                CHAT_AGENT_UNAVAILABLE,
                module="service.runtime",
                operation="answer",
                stage="validate_runtime",
                message="本地知识问答运行时尚未初始化",
            )

        try:
            invoke_kwargs: dict[str, object] = {}
            if self.checkpointer is not None:
                invoke_kwargs["config"] = self._thread_config(thread_id)
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                **invoke_kwargs,
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
        self.retriever = None
        self.retriever_tool = None
        self.agent = None
        self.checkpointer = None
        self.chat_model = None
        self.chunks = []
        self.chunk_count = 0

        if ingestion is not None:
            ingestion.close()

__all__ = ["Runtime"]
