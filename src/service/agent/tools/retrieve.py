"""把 Agent 的检索 Tool 转发给独立 vector-service。"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from src.errors import AppException, KNOWLEDGE_RETRIEVAL_FAILED


class VectorSearchClient(Protocol):
    """Agent 工具依赖的最小向量服务接口。"""

    def search_knowledge(self, queries: list[str]) -> str: ...

    def list_files(self) -> str: ...

    def find_file_by_name(self, filename: str) -> str: ...

    def read_knowledge_file(self, document_uuid: str) -> str: ...

    def read_credential_file(self, file_uuid: str) -> str: ...

    def search_credential_files(self, queries: list[str]) -> str: ...

    def search_bid_records(self, queries: list[str]) -> str: ...

    def search_credential_files_by_bid_record_id(
        self,
        bid_record_id: int,
        queries: list[str],
    ) -> str: ...

    def get_bid_record(self, bid_record_id: int) -> str: ...

    def list_bid_records(self) -> str: ...


class RetrieveInput(BaseModel):
    """要求模型把同一轮的全部检索子问题放入一个列表。"""

    queries: list[str] = Field(
        min_length=1,
        description="需要同时检索的全部独立问题；同一轮只提交一个 queries 列表。",
    )

    @field_validator("queries")
    @classmethod
    def normalize_queries(cls, queries: list[str]) -> list[str]:
        return _normalize_queries(queries)


class BidRecordInput(BaseModel):
    bid_record_id: int = Field(
        gt=0,
        description=(
            "仅当用户明确写出‘投标记录 13’这类纯数字记录编号时传入；"
            "不得从 PUBLIC-E2E-8047 等知识编号、文件编号或其他字符串的数字部分推导。"
        ),
    )


class FileLookupInput(BaseModel):
    filename: str = Field(
        min_length=1,
        description="用户提到的原文件名，只传文件名本身，不要传整句问题。",
    )

    @field_validator("filename")
    @classmethod
    def normalize_filename(cls, filename: str) -> str:
        normalized = filename.strip()
        if not normalized:
            raise ValueError("filename 不能为空")
        return normalized


class KnowledgeFileInput(BaseModel):
    document_uuid: str = Field(
        description="find_file_by_name 返回的知识文件 document_uuid。",
    )

    @field_validator("document_uuid")
    @classmethod
    def validate_document_uuid(cls, value: str) -> str:
        return _normalize_uuid(value, "document_uuid")


class CredentialFileInput(BaseModel):
    file_uuid: str = Field(
        description="find_file_by_name 返回的资信文件 file_uuid。",
    )

    @field_validator("file_uuid")
    @classmethod
    def validate_file_uuid(cls, value: str) -> str:
        return _normalize_uuid(value, "file_uuid")


class BidCredentialRetrieveInput(RetrieveInput):
    bid_record_id: int = Field(
        gt=0,
        description=(
            "已由用户明确写出或 get_bid_record 确认的投标记录 ID；"
            "不得从公共知识编号、文件编号或其他字符串的数字部分推导。"
        ),
    )


@dataclass(frozen=True, slots=True)
class RetrievalToolContext:
    """向 Agent Tool 注入独立向量服务客户端。"""

    vector_client: VectorSearchClient


@tool(args_schema=RetrieveInput)
def retrieve_knowledge_files(
    queries: list[str],
    runtime: ToolRuntime[RetrievalToolContext],
) -> str:
    """批量检索公共政策知识库，不会访问投标方私有资料。

    含字母和连字符的公共知识编号（如 ``PUBLIC-E2E-8047``）必须在此检索，
    其数字尾缀不是投标记录 ID。
    """

    normalized = _normalize_queries(queries)
    return _invoke(
        "retrieve_knowledge_files",
        lambda: runtime.context.vector_client.search_knowledge(normalized),
    )


@tool
def list_files(runtime: ToolRuntime[Any]) -> str:
    """列出全部已向量化的知识文件和资信文件；用户询问有哪些文件时调用。"""

    return _invoke(
        "list_files",
        runtime.context.vector_client.list_files,
    )


@tool(args_schema=FileLookupInput)
def find_file_by_name(
    filename: str,
    runtime: ToolRuntime[RetrievalToolContext],
) -> str:
    """按原文件名跨知识库和资信材料查找文件。

    用户提到具体文件名、询问某个文件内容或文件名可能有轻微拼写偏差时，必须先调用
    本工具；不要在未查询前声称无法访问文件。返回结果会给出文件类型和稳定 UUID。
    """

    normalized = filename.strip()
    return _invoke(
        "find_file_by_name",
        lambda: runtime.context.vector_client.find_file_by_name(normalized),
    )


@tool(args_schema=KnowledgeFileInput)
def read_knowledge_file(
    document_uuid: str,
    runtime: ToolRuntime[RetrievalToolContext],
) -> str:
    """读取 find_file_by_name 已确定的知识文件全部正文片段。"""

    normalized = _normalize_uuid(document_uuid, "document_uuid")
    return _invoke(
        "read_knowledge_file",
        lambda: runtime.context.vector_client.read_knowledge_file(normalized),
    )


@tool(args_schema=CredentialFileInput)
def read_credential_file(
    file_uuid: str,
    runtime: ToolRuntime[RetrievalToolContext],
) -> str:
    """读取已确定的资信文件正文及其关联投标记录，供综合分析使用。"""

    normalized = _normalize_uuid(file_uuid, "file_uuid")
    return _invoke(
        "read_credential_file",
        lambda: runtime.context.vector_client.read_credential_file(normalized),
    )


@tool(args_schema=RetrieveInput)
def retrieve_credential_files(
    queries: list[str],
    runtime: ToolRuntime[RetrievalToolContext],
) -> str:
    """跨全部投标记录语义检索资信文件正文。

    只用于资质、证书、附件等资信材料问题；公共知识编号不得调用本工具。
    """

    normalized = _normalize_queries(queries)
    return _invoke(
        "retrieve_credential_files",
        lambda: runtime.context.vector_client.search_credential_files(normalized),
    )


@tool(args_schema=RetrieveInput)
def retrieve_bid_records(
    queries: list[str],
    runtime: ToolRuntime[RetrievalToolContext],
) -> str:
    """跨全部投标记录检索投标人、报价、项目、公告和风险等结构化事实。"""

    normalized = _normalize_queries(queries)
    return _invoke(
        "retrieve_bid_records",
        lambda: runtime.context.vector_client.search_bid_records(normalized),
    )


@tool(args_schema=BidCredentialRetrieveInput)
def retrieve_credential_files_by_bid_record_id(
    bid_record_id: int,
    queries: list[str],
    runtime: ToolRuntime[RetrievalToolContext],
) -> str:
    """在一条已明确的投标记录内语义检索其资信文件正文。"""

    normalized = _normalize_queries(queries)
    return _invoke(
        "retrieve_credential_files_by_bid_record_id",
        lambda: runtime.context.vector_client.search_credential_files_by_bid_record_id(
            bid_record_id,
            normalized,
        ),
    )


@tool(args_schema=BidRecordInput)
def get_bid_record(
    bid_record_id: int,
    runtime: ToolRuntime[RetrievalToolContext],
) -> str:
    """按 ID 精确读取一条完整投标记录及其关联招标信息。"""

    return _invoke(
        "get_bid_record",
        lambda: runtime.context.vector_client.get_bid_record(bid_record_id),
    )


@tool
def list_bid_records(runtime: ToolRuntime[Any]) -> str:
    """列出向量服务中的全部投标记录摘要，供模型动态确定记录 ID。"""

    return _invoke(
        "list_bid_records",
        runtime.context.vector_client.list_bid_records,
    )


def _normalize_queries(queries: list[str]) -> list[str]:
    if isinstance(queries, (str, bytes)) or not queries:
        raise _error("validate_queries", "批量检索至少需要一个查询")
    normalized: list[str] = []
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise _error("validate_queries", "批量检索查询不能为空")
        normalized.append(query.strip())
    return normalized


def _normalize_uuid(value: str, field_name: str) -> str:
    try:
        return str(UUID(value.strip()))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是有效 UUID") from exc


def _invoke(tool_name: str, call: Callable[[], str]) -> str:
    try:
        result = call()
    except AppException:
        raise
    except Exception as exc:
        raise _error(
            "invoke_vector_service",
            "独立向量服务检索失败",
        ) from exc
    if not isinstance(result, str) or not result.strip():
        raise _error("validate_response", "独立向量服务返回了空检索结果")
    print(f"模型调用工具：{tool_name}", flush=True)
    return result


def _error(stage: str, message: str) -> AppException:
    return AppException(
        KNOWLEDGE_RETRIEVAL_FAILED,
        module="service.agent.tools.retrieve",
        operation="retrieve",
        stage=stage,
        message=message,
    )


__all__ = [
    "BidCredentialRetrieveInput",
    "BidRecordInput",
    "CredentialFileInput",
    "FileLookupInput",
    "KnowledgeFileInput",
    "RetrieveInput",
    "RetrievalToolContext",
    "VectorSearchClient",
    "find_file_by_name",
    "get_bid_record",
    "list_files",
    "list_bid_records",
    "read_credential_file",
    "read_knowledge_file",
    "retrieve_bid_records",
    "retrieve_credential_files_by_bid_record_id",
    "retrieve_credential_files",
    "retrieve_knowledge_files",
]
