"""应用级错误类型的统一出口。

本包对外提供 ``AppErrorCode`` 和 ``AppException``，以及按业务模块归并的
错误代码常量。业务模块通过这里导入错误对象并抛出结构化异常；上层
服务再根据异常中的错误代码、阶段和 details 记录日志、更新任务状态或生成
HTTP 响应。本包不负责执行具体业务，也不负责主动处理异常。
"""

from .exceptions import (
    AppErrorCode,
    AppException,
    CHAT_AGENT_FAILED,
    CHAT_AGENT_UNAVAILABLE,
    CHAT_INPUT_INVALID,
    KNOWLEDGE_DOCUMENT_LOAD_FAILED,
    KNOWLEDGE_RETRIEVAL_FAILED,
)

__all__ = [
    "AppErrorCode",
    "AppException",
    "CHAT_AGENT_FAILED",
    "CHAT_AGENT_UNAVAILABLE",
    "CHAT_INPUT_INVALID",
    "KNOWLEDGE_DOCUMENT_LOAD_FAILED",
    "KNOWLEDGE_RETRIEVAL_FAILED",
]
