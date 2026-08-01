"""管理本地知识问答服务运行期间的模型、向量库和 LangChain Agent。

``Runtime`` 在服务启动时一次性创建 Embedding 模型、持久化 Chroma、检索器、
ChatOpenAI 客户端和 LangChain Agent；请求处理只复用这些对象。当前运行时不
保存对话历史，每次 ``answer`` 只使用当前问题和当前检索结果。服务关闭时由
``close`` 统一释放 Chroma、Agent、模型引用和设备缓存。
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from configuration import (
    ChatModelSettings,
    KnowledgeRuntimeSettings,
    RetrievalSettings,
    get_chat_model_settings,
    get_knowledge_runtime_settings,
    get_retrieval_settings,
)

from src.errors import (
    AppException,
    CHAT_AGENT_FAILED,
    CHAT_AGENT_UNAVAILABLE,
    CHAT_INPUT_INVALID,
)

from .agent import LangChainChatAgent
from .ingestion import IngestionService
from .retrieval import VectorRetriever


class Runtime:
    """持有单进程知识问答服务的全部运行时资源。"""

    def __init__(
        self,
        settings: KnowledgeRuntimeSettings | None = None,
        retrieval_settings: RetrievalSettings | None = None,
    ) -> None:
        """加载 Embedding、持久化 Chroma、政策数据和无状态 Agent。"""

        self.settings = settings or get_knowledge_runtime_settings()
        self.retrieval_settings = retrieval_settings or get_retrieval_settings()
        self.ingestion: IngestionService | None = None
        self.retriever: VectorRetriever | None = None
        self.chat_model: ChatOpenAI | None = None
        self.agent: LangChainChatAgent | None = None
        self.chunks: list[Any] = []
        self.chunk_count = 0

        try:
            print("正在加载本地 Embedding 模型和持久化 Chroma...", flush=True)
            self.ingestion = IngestionService(
                model_path=self.settings.embedding_model_path,
                chroma_path=self.settings.chroma_path,
            )
            self.retriever = VectorRetriever(self.ingestion)

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
            self.chat_model = self._create_chat_model(chat_settings)
            self.agent = LangChainChatAgent(
                chat_model=self.chat_model,
                settings=chat_settings,
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

    def answer(self, question: str) -> str:
        """独立处理一个问题，不读取或保存其他请求的上下文历史。"""

        if not isinstance(question, str) or not question.strip():
            raise AppException(
                CHAT_INPUT_INVALID,
                module="service.runtime",
                operation="answer",
                stage="validate_input",
                message="问题不能为空",
            )
        if self.retriever is None or self.agent is None:
            raise AppException(
                CHAT_AGENT_UNAVAILABLE,
                module="service.runtime",
                operation="answer",
                stage="validate_runtime",
                message="本地知识问答运行时尚未初始化",
            )

        try:
            results = self.retriever.retrieve(
                question,
                top_k=self.retrieval_settings.top_k,
            )
            context = self._format_context(results)
            answer = self.agent.complete(
                question,
                context=context,
            )
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

    def close(self) -> None:
        """释放 Agent、模型、Chroma 客户端和设备缓存。"""

        ingestion = self.ingestion
        self.ingestion = None
        self.retriever = None
        self.agent = None
        self.chat_model = None
        self.chunks = []
        self.chunk_count = 0

        if ingestion is not None:
            ingestion.close()

    @staticmethod
    def _create_chat_model(settings: ChatModelSettings) -> ChatOpenAI:
        """按配置创建本地 OpenAI-compatible ChatOpenAI 客户端。"""

        return ChatOpenAI(
            model=settings.model_name,
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_completion_tokens=settings.max_tokens,
            extra_body={
                "top_k": settings.top_k,
                "repetition_penalty": settings.repetition_penalty,
            },
        )

    @staticmethod
    def _format_context(results: dict[str, Any]) -> str:
        """把 Chroma 返回的 Top-K 片段整理为 Prompt 上下文。"""

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]
        document_list = documents[0] if documents else []
        metadata_list = metadatas[0] if metadatas else []
        distance_list = distances[0] if distances else []

        if not document_list:
            return "未检索到相关政策片段。"

        context: list[str] = []
        for index, document in enumerate(document_list):
            metadata = metadata_list[index] if index < len(metadata_list) else {}
            distance = distance_list[index] if index < len(distance_list) else "未知"
            context.append(
                "\n".join(
                    [
                        f"[资料片段 {index + 1}]",
                        f"标题：{metadata.get('title', '未知')}",
                        f"日期：{metadata.get('date', '未知')}",
                        f"来源：{metadata.get('source', '未知')}",
                        f"相似度距离：{distance}",
                        f"正文：\n{document}",
                    ]
                )
            )
        return "\n\n---\n\n".join(context)


__all__ = ["Runtime"]
