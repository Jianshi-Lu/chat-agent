"""实现无状态对话机器人的外部调用入口。

调用方通过 ``ChatClient.complete`` 或 ``ChatClient.stream_response`` 提交一条用户消息；本模块
将消息交给基于 LangChain 的 ``LangChainChatAgent``，由它调用 OMLX 提供的
OpenAI-compatible 服务。每次请求彼此独立，不维护上下文，也不执行检索或工具编排。
"""

from collections.abc import Iterator

from app.errors import AppException, CHAT_AGENT_FAILED, CHAT_AGENT_UNAVAILABLE
from app.agent.agent import LangChainChatAgent


class ChatClient:
    """接收单条用户消息并转发给无状态对话 Agent。

    输入：用户消息字符串；调用方可以选择同步获取完整回答，或通过 ``stream_response``
    获取文本片段。

    处理：入口只负责默认 Agent 创建、调用转发和异常包装；输入校验由后续独立的
    校验模块负责。

    输出：``complete`` 返回完整字符串；``stream_response`` 返回按 Agent 顺序产生的
    字符串迭代器。
    """

    def __init__(self) -> None:
        """创建对话入口。

        Agent 和模型由 ``LangChainChatAgent`` 按 ``configuration`` 中的配置创建。
        """

        try:
            self._agent = LangChainChatAgent()
        except Exception as exc:
            raise AppException(
                CHAT_AGENT_UNAVAILABLE,
                module="service.client",
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
                module="service.client",
                operation="complete",
                stage="agent_complete",
                message="对话 Agent 调用失败",
            ) from exc

        if not isinstance(response, str):
            raise AppException(
                CHAT_AGENT_FAILED,
                module="service.client",
                operation="complete",
                stage="validate_agent_response",
                message="对话 Agent 返回格式无效",
                details={"expected_type": "str"},
            )

        return response

    def stream_response(
        self,
        message: str,
    ) -> Iterator[str]:
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
                module="service.client",
                operation="stream_response",
                stage="agent_stream_response",
                message="对话 Agent 流式调用失败",
            ) from exc


__all__ = ["ChatClient"]
