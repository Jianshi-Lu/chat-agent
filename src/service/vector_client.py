"""chat-agent 访问独立 vector-service 的同步 HTTP 客户端。"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from configuration import VectorServiceSettings
from src.errors import (
    AppException,
    CHAT_AGENT_UNAVAILABLE,
    KNOWLEDGE_RETRIEVAL_FAILED,
)


_INTERNAL_SOURCE_LINE_PATTERN = re.compile(
    r"(?m)^来源：[^\r\n]*(?:\r?\n|$)"
)
_INTERNAL_CREDENTIAL_OBJECT_PATH_SECTION_PATTERN = re.compile(
    r"(?ms)^##\s*资信材料对象路径\s*\r?\n.*?(?=^##\s|\Z)"
)
_INTERNAL_FILE_ID_LINE_PATTERN = re.compile(
    r"(?mi)^(?:file_uuid|document_uuid)[：:][^\r\n]*(?:\r?\n|$)"
)
_PUBLIC_KNOWLEDGE_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:PUBLIC|POLICY|KNOWLEDGE)(?:[-_][A-Z0-9]+)+\b", re.IGNORECASE
)
_SEARCH_QUERY_HEADER_PATTERN = re.compile(r"(?m)^查询\s*\d+[：:][^\r\n]*(?:\r?\n|$)")


class VectorServiceClient:
    """封装 Agent 工具所需的最小向量检索协议。"""

    def __init__(self, settings: VectorServiceSettings) -> None:
        self.base_url = settings.base_url.rstrip("/")
        self.timeout = settings.timeout

    def health(self) -> None:
        """在 Agent 启动前确认 vector-service 已完成资源初始化。"""

        try:
            payload = self._request("GET", "/health")
        except AppException as exc:
            raise AppException(
                CHAT_AGENT_UNAVAILABLE,
                module="service.vector_client",
                operation="health",
                stage="connect_vector_service",
                message="独立向量服务不可用",
                details={"base_url": self.base_url},
            ) from exc
        if payload.get("status") != "ready":
            raise AppException(
                CHAT_AGENT_UNAVAILABLE,
                module="service.vector_client",
                operation="health",
                stage="validate_vector_service",
                message="独立向量服务尚未就绪",
                details={"base_url": self.base_url},
            )

    def search_knowledge(self, queries: list[str]) -> str:
        response = self._search("/internal/search/knowledge", queries)
        # 相似度检索在没有相关文档时仍会返回 top-k。对于带有稳定公共知识编号的
        # 查询，若结果中完全不含该编号，不能把任意相似文本交给模型生成答案。
        # 这样会把“编号已删除/未入库”显式交给 Agent，而不是制造貌似可信的回答。
        identifiers = {
            match.group(0).upper()
            for query in queries
            for match in _PUBLIC_KNOWLEDGE_IDENTIFIER_PATTERN.finditer(query)
        }
        # ``_retrieve_queries`` 会原样回显“查询 N：<query>”；那不是命中文档的
        # 证据，必须从比对文本中排除，否则任何编号查询都会错误通过此校验。
        evidence = _SEARCH_QUERY_HEADER_PATTERN.sub("", response)
        if identifiers and not any(
            identifier.casefold() in evidence.casefold()
            for identifier in identifiers
        ):
            quoted_identifiers = "、".join(
                f"“{identifier}”" for identifier in sorted(identifiers)
            )
            return f"未找到包含公共知识编号{quoted_identifiers}的当前已入库内容。"
        return response

    def list_ingested_files(self) -> str:
        return self._response_text(
            self._request("GET", "/internal/search/files"),
            preserve_file_identifiers=True,
        )

    def find_ingested_files(self, filename: str) -> str:
        return self._response_text(
            self._request("POST", "/internal/search/files", {"filename": filename}),
            preserve_file_identifiers=True,
        )

    def read_knowledge_file(self, document_uuid: str) -> str:
        return self._response_text(
            self._request(
                "GET",
                f"/internal/search/knowledge-files/{document_uuid}",
            )
        )

    def read_credential_file(self, file_uuid: str) -> str:
        return self._response_text(
            self._request(
                "GET",
                f"/internal/search/credential-files/{file_uuid}",
            )
        )

    def search_bid_materials(self, queries: list[str]) -> str:
        return self._search("/internal/search/bid-materials", queries)

    def search_bid_credentials(
        self,
        bid_record_id: int,
        queries: list[str],
    ) -> str:
        return self._search(
            f"/internal/search/bid-records/{bid_record_id}/credentials",
            queries,
        )

    def get_bid_record(self, bid_record_id: int) -> str:
        return self._response_text(
            self._request(
                "GET",
                f"/internal/search/bid-records/{bid_record_id}",
            )
        )

    def list_bid_records(self) -> str:
        return self._response_text(
            self._request("GET", "/internal/search/bid-records")
        )

    def close(self) -> None:
        """urllib 不持有跨请求连接；保留统一资源释放接口。"""

    def _search(self, path: str, queries: list[str]) -> str:
        return self._response_text(
            self._request("POST", path, {"queries": queries})
        )

    @staticmethod
    def _response_text(
        payload: dict[str, Any],
        *,
        preserve_file_identifiers: bool = False,
    ) -> str:
        value = payload.get("text")
        if not isinstance(value, str) or not value.strip():
            raise AppException(
                KNOWLEDGE_RETRIEVAL_FAILED,
                module="service.vector_client",
                operation="parse_response",
                stage="validate_response",
                message="独立向量服务返回了无效检索结果",
            )
        # Chroma 的 ``source`` metadata、资信对象路径和文件 UUID 只用于服务间
        # 追溯。它们不能被模型当成面向用户的下载链接；真实下载必须经 Java 授权接口
        # 和前端卡片。按文件名查找时仍保留 UUID，供模型继续调用 read_* Tool。
        sanitized = _INTERNAL_SOURCE_LINE_PATTERN.sub("", value)
        sanitized = _INTERNAL_CREDENTIAL_OBJECT_PATH_SECTION_PATTERN.sub("", sanitized)
        if not preserve_file_identifiers:
            sanitized = _INTERNAL_FILE_ID_LINE_PATTERN.sub("", sanitized)
        return sanitized.strip()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise self._request_error(
                stage="vector_service_http_error",
                message=f"独立向量服务返回 HTTP {exc.code}",
                path=path,
            ) from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise self._request_error(
                stage="connect_vector_service",
                message="无法连接独立向量服务",
                path=path,
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise self._request_error(
                stage="decode_vector_service_response",
                message="独立向量服务返回了无效 JSON",
                path=path,
            ) from exc
        if not isinstance(decoded, dict):
            raise self._request_error(
                stage="validate_vector_service_response",
                message="独立向量服务响应必须是 JSON 对象",
                path=path,
            )
        return decoded

    @staticmethod
    def _request_error(*, stage: str, message: str, path: str) -> AppException:
        return AppException(
            KNOWLEDGE_RETRIEVAL_FAILED,
            module="service.vector_client",
            operation="request",
            stage=stage,
            message=message,
            details={"path": path},
        )


__all__ = ["VectorServiceClient"]
