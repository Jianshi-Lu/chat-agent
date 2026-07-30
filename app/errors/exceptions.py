"""定义应用级错误代码和结构化异常。

业务模块在执行文件读取、模型校验等步骤时，使用本模块的错误代码构造
``AppException``。异常向上流转时携带稳定的错误编码、调用位置、处理阶段和
安全的结构化细节；上层任务或 API 边界再根据这些信息更新状态或生成响应。
底层异常应使用 ``raise ... from exc`` 作为原因保留下来，便于定位真实故障。
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AppErrorCode:
    """描述一种可跨服务识别的应用错误。

    字段输入由代码常量提供：``code`` 是 Python、Java 和监控系统共享的稳定
    编码，``http_status`` 是 API 层可使用的状态码，``default_message`` 是
    无自定义消息时的默认提示，``retryable`` 表示上层是否可以安排重试。
    ``frozen=True`` 和 ``slots=True`` 使错误代码对象不可变且结构固定；业务
    模块只引用既有对象，不在运行时修改或重新解释其含义。
    """

    code: str
    http_status: int
    default_message: str
    retryable: bool


class AppException(Exception):
    """承载错误语义和调用上下文的应用异常。

    业务代码输入一个 ``AppErrorCode``，可选输入模块名、操作名、处理阶段、
    面向调用方的消息和结构化 details。构造过程会把自定义消息或错误代码的
    默认消息交给 ``Exception``，同时保存完整上下文；上层通过异常属性读取
    这些信息。该类不负责补偿、重试或生成 HTTP 响应，原始底层异常由调用方
    使用 ``raise ... from exc`` 建立异常链。
    """

    def __init__(
        self,
        error_code: AppErrorCode,
        *,
        module: str | None = None,
        operation: str | None = None,
        stage: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """初始化结构化异常。

        参数：
            error_code: 当前错误对应的稳定错误代码。
            module: 发生错误的模块名，例如 ``knowledge.loader``。
            operation: 发生错误的公开操作名。
            stage: 操作链中的具体阶段，例如 ``parse_json``。
            message: 可安全展示给调用方的自定义消息；为空时使用默认消息。
            details: 可安全记录的附加字段，例如文档索引或文件路径。

        返回：
            无返回值；异常实例保存错误代码和上下文，等待上层捕获处理。

        逻辑：
            先选择自定义消息或默认消息，再复制 ``details``，避免调用方后续
            修改原字典时影响异常实例中的错误上下文。
        """

        public_message = message or error_code.default_message
        super().__init__(public_message)

        self.error_code = error_code
        self.module = module
        self.operation = operation
        self.stage = stage
        self.public_message = public_message
        self.details = dict(details or {})


# 业务模块按模块粒度复用错误代码；具体失败逻辑通过 AppException 的
# ``message``、``stage`` 和 ``details`` 描述。
KNOWLEDGE_DOCUMENT_LOAD_FAILED = AppErrorCode(
    code="KNOWLEDGE_INGESTION_AGENT.DOCUMENT_LOAD_FAILED",
    http_status=500,
    default_message="知识文档加载失败",
    retryable=False,
)

KNOWLEDGE_RETRIEVAL_FAILED = AppErrorCode(
    code="KNOWLEDGE_INGESTION_AGENT.RETRIEVAL_FAILED",
    http_status=500,
    default_message="知识文档检索失败",
    retryable=False,
)
