"""实现基于 LangChain 官方 ``create_agent`` 的无状态本地对话 Agent。

本模块使用 ``ChatOpenAI`` 连接本机 OMLX 服务，并把 ``src.agent.tools`` 中的工具
交给 ``create_agent``。工具选择、工具执行、结果回传和最终回答由官方 Agent
运行时编排；本模块每次只传入当前用户消息，不配置 ``checkpointer``，因此不保存
跨请求上下文。
"""

from collections.abc import Iterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from configuration import ChatSettings, get_chat_settings

from .tools import exact_paragraph_search, semantic_search


_MODEL_NODE = "model"


class LangChainChatAgent:
    """通过 LangChain 官方 Agent 调用 OMLX 模型且不维护上下文。"""

    settings: ChatSettings
    model: ChatOpenAI
    agent: Runnable

    def __init__(self) -> None:
        """创建模型适配器。

        配置从根路径的 ``configuration`` 包读取，模型和官方 Agent 都由本模块
        按配置创建。
        """

        self.settings: ChatSettings = get_chat_settings()
        self.model: ChatOpenAI = self._create_model(self.settings)
        self.agent: Runnable = create_agent(
            model=self.model,
            tools=[exact_paragraph_search, semantic_search],
            system_prompt=self.settings.system_prompt,
            name="local_chat_agent",
        )

    def complete(self, message: str) -> str:
        """使用官方 Agent 处理当前消息并一次性返回最终回答。"""

        result = self.agent.invoke({"messages": self._build_messages(message)})
        return self._result_text(result)

    def stream_response(self, message: str) -> Iterator[str]:
        """使用官方 Agent 流式返回最终回答文本片段。"""

        for chunk in self.agent.stream(
            {"messages": self._build_messages(message)},
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

    def _build_messages(self, message: str) -> list[BaseMessage]:
        """为单次 Agent 调用构造消息，不读取或修改历史状态。"""

        return [HumanMessage(content=message)]

    @staticmethod
    def _result_text(result: object) -> str:
        """从官方 Agent 的状态结果中提取最后一条消息文本。"""

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
    def _create_model(settings: ChatSettings) -> ChatOpenAI:
        """按本地服务配置创建 ``ChatOpenAI`` 客户端。"""

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
