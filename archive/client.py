"""归档的无状态对话调用入口。

调用方通过 ``ChatClient.complete`` 或 ``ChatClient.stream_response`` 提交一条
用户消息；本模块将消息交给 ``LangChainChatAgent``，由它调用本地
OpenAI-compatible 服务。每次请求彼此独立，不维护上下文，也不执行知识库
检索或 RAG 编排。当前主流程使用 ``src.service.__main__``，本模块仅保留
早期通用聊天客户端实现。
"""

from collections.abc import Iterator

from src.agent.agent import LangChainChatAgent
from src.errors import AppException, CHAT_AGENT_FAILED, CHAT_AGENT_UNAVAILABLE


class ChatClient:
    """接收单条用户消息并转发给无状态对话 Agent。"""

    def __init__(self) -> None:
        """创建由配置驱动的无状态对话 Agent。"""

        try:
            self._agent = LangChainChatAgent()
        except Exception as exc:
            raise AppException(
                CHAT_AGENT_UNAVAILABLE,
                module="archive.client",
                operation="create_agent",
                stage="initialize_agent",
                message="对话 Agent 初始化失败",
            ) from exc

    def complete(self, message: str) -> str:
        """提交用户消息并一次性返回完整回答。"""

        try:
            response = self._agent.complete(message)
        except AppException:
            raise
        except Exception as exc:
            raise AppException(
                CHAT_AGENT_FAILED,
                module="archive.client",
                operation="complete",
                stage="agent_complete",
                message="对话 Agent 调用失败",
            ) from exc

        if not isinstance(response, str):
            raise AppException(
                CHAT_AGENT_FAILED,
                module="archive.client",
                operation="complete",
                stage="validate_agent_response",
                message="对话 Agent 返回格式无效",
                details={"expected_type": "str"},
            )

        return response

    def stream_response(self, message: str) -> Iterator[str]:
        """提交用户消息并转发 Agent 产生的回答片段。"""

        try:
            for chunk in self._agent.stream_response(message):
                if not isinstance(chunk, str):
                    raise TypeError("Agent stream chunk must be str")
                if chunk:
                    yield chunk
        except AppException:
            raise
        except Exception as exc:
            raise AppException(
                CHAT_AGENT_FAILED,
                module="archive.client",
                operation="stream_response",
                stage="agent_stream_response",
                message="对话 Agent 流式调用失败",
            ) from exc


__all__ = ["ChatClient"]
