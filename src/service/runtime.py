"""管理本地知识问答服务运行期间的模型、向量库和 RAG 链。

``Runtime`` 在服务启动时一次性创建 Embedding 模型、内存 Chroma、检索器和
LangChain RAG Chain；请求处理只复用这些对象。当前运行时不保存对话历史，
每次 ``ask`` 只使用当前问题和当前检索结果。服务关闭时由 ``close`` 统一释放
Chroma、模型引用和设备缓存。
"""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from configuration import ChatSettings, ServerSettings, get_chat_settings, get_server_settings

from src.errors import (
    AppException,
    CHAT_AGENT_FAILED,
    CHAT_AGENT_UNAVAILABLE,
    CHAT_INPUT_INVALID,
)

from .ingestion import IngestionService
from .retrieval import VectorRetriever


class Runtime:
    """持有单进程知识问答服务的全部运行时资源。"""

    def __init__(self, settings: ServerSettings | None = None) -> None:
        """加载 Embedding、内存 Chroma、政策数据和无状态 RAG Chain。"""

        self.settings = settings or get_server_settings()
        self.ingestion: IngestionService | None = None
        self.retriever: VectorRetriever | None = None
        self.model: Runnable | None = None
        self.chain: Runnable | None = None
        self.chunks: list[Any] = []

        try:
            print("正在加载本地 Embedding 模型和内存 Chroma...", flush=True)
            self.ingestion = IngestionService(
                model_path=self.settings.embedding_model_path,
            )
            self.retriever = VectorRetriever(self.ingestion)

            print("正在读取 policies.json 并生成文档向量...", flush=True)
            self.chunks = self.ingestion.ingest(self.settings.policies_path)

            print("正在初始化 LangChain 本地对话链...", flush=True)
            chat_settings = get_chat_settings()
            self.model = self._create_model(chat_settings)
            self.chain = self._create_chain()
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

    def ask(self, question: str) -> str:
        """独立处理一个问题，不读取或保存其他请求的上下文历史。"""

        if not isinstance(question, str) or not question.strip():
            raise AppException(
                CHAT_INPUT_INVALID,
                module="service.runtime",
                operation="ask",
                stage="validate_input",
                message="问题不能为空",
            )
        if self.retriever is None or self.chain is None:
            raise AppException(
                CHAT_AGENT_UNAVAILABLE,
                module="service.runtime",
                operation="ask",
                stage="validate_runtime",
                message="本地知识问答运行时尚未初始化",
            )

        try:
            results = self.retriever.retrieve(
                question,
                top_k=self.settings.top_k,
            )
            context = self._format_context(results)
            answer = self.chain.invoke(
                {
                    "context": context,
                    "question": question,
                }
            )
        except AppException:
            raise
        except Exception as exc:
            raise AppException(
                CHAT_AGENT_FAILED,
                module="service.runtime",
                operation="ask",
                stage="invoke_chain",
                message="本地 RAG Chain 调用失败",
            ) from exc

        if not isinstance(answer, str) or not answer.strip():
            raise AppException(
                CHAT_AGENT_FAILED,
                module="service.runtime",
                operation="ask",
                stage="validate_response",
                message="本地 RAG Chain 返回了空回答",
            )
        return answer

    def close(self) -> None:
        """释放 RAG Chain、模型、内存 Chroma 和设备缓存。"""

        ingestion = self.ingestion
        self.ingestion = None
        self.retriever = None
        self.chain = None
        self.model = None
        self.chunks = []

        if ingestion is not None:
            ingestion.close()

    def _create_chain(self) -> Runnable:
        """创建只接收当前问题和检索上下文的无状态 RAG Chain。"""

        system_prompt = (
            "你是政策资料问答助手。只能依据用户提供的知识库上下文回答问题；"
            "如果上下文不足以支持结论，必须明确说明知识库中没有足够信息，"
            "不要凭空补充事实。回答尽量简洁，并在适合时指出资料来源。"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "知识库上下文：\n{context}\n\n当前问题：\n{question}",
                ),
            ]
        )
        if self.model is None:
            raise RuntimeError("对话模型尚未初始化")
        return prompt | self.model | StrOutputParser()

    @staticmethod
    def _create_model(settings: ChatSettings) -> Runnable:
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
