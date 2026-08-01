"""实现服务主链路使用的无状态 LangChain Agent。

Agent 复用 Runtime 已创建的 ``ChatOpenAI`` 客户端，并注册本包中的工具。
每次调用只接收当前问题和本次检索得到的上下文，不配置 checkpointer，因而
不会保存跨 HTTP 请求的对话历史。
"""

from collections.abc import Iterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from configuration import ChatModelSettings, get_chat_model_settings

from .tools import exact_paragraph_search, semantic_search


_MODEL_NODE = "model"
_DEFAULT_SYSTEM_PROMPT = (
    "你是政策资料问答助手。只能依据用户消息中提供的知识库上下文回答问题；"
    "如果上下文不足以支持结论，必须明确说明知识库中没有足够信息，"
    "不要凭空补充事实。已注册的工具当前可能返回占位结果，不能把占位结果"
    "当作真实政策事实。回答尽量简洁，并在适合时指出资料来源。"
)


class LangChainChatAgent:
    """通过 LangChain 官方 Agent 处理单次无状态对话。"""

    settings: ChatModelSettings
    chat_model: ChatOpenAI
    runnable: Runnable

    def __init__(
        self,
        chat_model: ChatOpenAI | None = None,
        settings: ChatModelSettings | None = None,
    ) -> None:
        """创建 Agent，并可复用 Runtime 已创建的模型客户端。"""

        self.settings = settings or get_chat_model_settings()
        self.chat_model = chat_model or self._create_chat_model(self.settings)
        self.runnable = create_agent(
            model=self.chat_model,
            tools=[exact_paragraph_search, semantic_search],
            system_prompt=self.settings.system_prompt or _DEFAULT_SYSTEM_PROMPT,
            name="local_chat_agent",
        )

    def complete(self, message: str, *, context: str = "") -> str:
        """提交当前问题和上下文，并返回 Agent 的最终回答。"""

        result = self.runnable.invoke(
            {"messages": self._build_messages(message, context=context)}
        )
        return self._result_text(result)

    def stream_response(
        self,
        message: str,
        *,
        context: str = "",
    ) -> Iterator[str]:
        """流式返回当前问题对应的回答文本片段。"""

        for chunk in self.runnable.stream(
            {"messages": self._build_messages(message, context=context)},
            stream_mode="messages",
            version="v2",
        ):
            stream_message = self._stream_message(chunk)
            if stream_message is None:
                continue

            message_chunk, metadata = stream_message
            if metadata.get("langgraph_node") != _MODEL_NODE:
                continue
            if not isinstance(message_chunk, (AIMessage, AIMessageChunk)):
                continue
            if message_chunk.tool_calls or getattr(
                message_chunk,
                "tool_call_chunks",
                (),
            ):
                continue

            text = self._message_text(message_chunk)
            if text:
                yield text

    @staticmethod
    def _build_messages(message: str, *, context: str) -> list[BaseMessage]:
        """把当前问题和本次检索上下文组成一条用户消息。"""

        if not context:
            return [HumanMessage(content=message)]

        content = f"知识库上下文：\n{context or '未提供上下文。'}\n\n当前问题：\n{message}"
        return [HumanMessage(content=content)]

    @staticmethod
    def _result_text(result: object) -> str:
        """从 Agent 状态结果中提取最后一条模型消息文本。"""

        if not isinstance(result, dict):
            raise TypeError("Agent 返回结果必须是包含 messages 的字典")

        messages = result.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Agent 返回结果中缺少有效 messages")

        final_message = messages[-1]
        if not isinstance(final_message, (AIMessage, AIMessageChunk)):
            raise TypeError("Agent 最后一条消息不是模型回答")

        text = LangChainChatAgent._message_text(final_message)
        if not text:
            raise ValueError("Agent 最终回答不包含可用文本")
        return text

    @staticmethod
    def _stream_message(
        chunk: object,
    ) -> tuple[object, dict[str, object]] | None:
        """提取 v2 ``messages`` 事件中的消息和 LangGraph metadata。"""

        if not isinstance(chunk, dict) or chunk.get("type") != "messages":
            return None

        data = chunk.get("data")
        if not isinstance(data, tuple) or len(data) != 2:
            return None

        message, metadata = data
        if not isinstance(metadata, dict):
            return None

        return message, metadata

    @staticmethod
    def _message_text(message: object) -> str:
        """提取 LangChain 消息或流式消息片段中的文本内容。"""

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
    def _create_chat_model(settings: ChatModelSettings) -> ChatOpenAI:
        """按本地 OpenAI-compatible 服务配置创建模型客户端。"""

        return ChatOpenAI(
            model=settings.model_name,
            base_url=settings.base_url,
            api_key=settings.api_key,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_completion_tokens=settings.max_tokens,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
            extra_body={
                "top_k": settings.top_k,
                "repetition_penalty": settings.repetition_penalty,
            },
        )


__all__ = ["LangChainChatAgent"]
