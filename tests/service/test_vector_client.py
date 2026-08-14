"""验证 chat-agent 与独立 vector-service 的 HTTP 契约。"""

import json

import pytest

import src.service.vector_client as client_module
from configuration import VectorServiceSettings
from src.errors import AppException, CHAT_AGENT_UNAVAILABLE
from src.service.vector_client import VectorServiceClient


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_search_posts_queries_to_vector_service(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, *, timeout: float):
        captured["url"] = request.full_url
        captured["method"] = request.method
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse({"text": "检索结果"})

    monkeypatch.setattr(client_module, "urlopen", fake_urlopen)
    client = VectorServiceClient(
        VectorServiceSettings(base_url="http://127.0.0.1:8083", timeout=9)
    )

    assert client.search_knowledge(["问题一", "问题二"]) == "检索结果"
    assert captured == {
        "url": "http://127.0.0.1:8083/internal/search/knowledge",
        "method": "POST",
        "body": {"queries": ["问题一", "问题二"]},
        "timeout": 9,
    }


def test_search_public_identifier_rejects_unrelated_top_k_results(monkeypatch) -> None:
    monkeypatch.setattr(
        client_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            {"text": "查询 1：PUBLIC-E2E-8047 的保证金\n文件：通用说明.pdf\n正文：保证金说明"}
        ),
    )
    client = VectorServiceClient(VectorServiceSettings())

    assert client.search_knowledge(["PUBLIC-E2E-8047 的保证金"]) == (
        "未找到包含公共知识编号“PUBLIC-E2E-8047”的当前已入库内容。"
    )


def test_search_public_identifier_keeps_matching_result(monkeypatch) -> None:
    monkeypatch.setattr(
        client_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            {"text": "查询 1：PUBLIC-E2E-8047 的保证金\n文件：政策.pdf\n正文：PUBLIC-E2E-8047，保证金 88888 元"}
        ),
    )
    client = VectorServiceClient(VectorServiceSettings())

    assert client.search_knowledge(["PUBLIC-E2E-8047 的保证金"]) == (
        "查询 1：PUBLIC-E2E-8047 的保证金\n文件：政策.pdf\n正文：PUBLIC-E2E-8047，保证金 88888 元"
    )


def test_search_hides_internal_object_storage_source(monkeypatch) -> None:
    monkeypatch.setattr(
        client_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            {"text": "文件：example.pdf\n来源：bid-records/13/converted/example.md\n正文：内容"}
        ),
    )
    client = VectorServiceClient(VectorServiceSettings())

    assert client.search_credential_files(["示例"]) == "文件：example.pdf\n正文：内容"


def test_credential_search_uses_renamed_vector_service_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, *, timeout: float):
        captured["url"] = request.full_url
        captured["method"] = request.method
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse({"text": "检索结果"})

    monkeypatch.setattr(client_module, "urlopen", fake_urlopen)
    client = VectorServiceClient(
        VectorServiceSettings(base_url="http://127.0.0.1:8083", timeout=9)
    )

    assert client.search_credential_files(["资信"]) == "检索结果"
    assert captured == {
        "url": "http://127.0.0.1:8083/internal/search/credential-files",
        "method": "POST",
        "body": {"queries": ["资信"]},
        "timeout": 9,
    }


def test_read_credential_hides_internal_object_path_and_file_uuid(monkeypatch) -> None:
    monkeypatch.setattr(
        client_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            {
                "text": (
                    "资信文件：example.pdf\n"
                    "file_uuid：550e8400-e29b-41d4-a716-446655440000\n"
                    "正文：内容\n\n"
                    "## 资信材料对象路径\n\n"
                    "- bidder/example.pdf/converted/example.md\n\n"
                    "## 招标公告\n\n- 项目编号：E2E"
                )
            }
        ),
    )
    client = VectorServiceClient(VectorServiceSettings())

    assert client.read_credential_file(
        "550e8400-e29b-41d4-a716-446655440000"
    ) == "资信文件：example.pdf\n正文：内容\n\n## 招标公告\n\n- 项目编号：E2E"


def test_file_tools_use_exact_vector_service_endpoints(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, *, timeout: float):
        captured.append(
            {
                "url": request.full_url,
                "method": request.method,
                "body": (
                    json.loads(request.data.decode("utf-8"))
                    if request.data is not None
                    else None
                ),
                "timeout": timeout,
            }
        )
        return FakeResponse({"text": "文件结果"})

    monkeypatch.setattr(client_module, "urlopen", fake_urlopen)
    client = VectorServiceClient(
        VectorServiceSettings(base_url="http://127.0.0.1:8083", timeout=9)
    )
    document_uuid = "550e8400-e29b-41d4-a716-446655440000"
    file_uuid = "550e8400-e29b-41d4-a716-446655440001"

    assert client.list_files() == "文件结果"
    assert client.find_file_by_name("complex.pdf") == "文件结果"
    assert client.read_knowledge_file(document_uuid) == "文件结果"
    assert client.read_credential_file(file_uuid) == "文件结果"

    assert captured == [
        {
            "url": "http://127.0.0.1:8083/internal/search/files",
            "method": "GET",
            "body": None,
            "timeout": 9,
        },
        {
            "url": "http://127.0.0.1:8083/internal/search/files",
            "method": "POST",
            "body": {"filename": "complex.pdf"},
            "timeout": 9,
        },
        {
            "url": (
                "http://127.0.0.1:8083/internal/search/knowledge-files/"
                + document_uuid
            ),
            "method": "GET",
            "body": None,
            "timeout": 9,
        },
        {
            "url": (
                "http://127.0.0.1:8083/internal/search/credential-files/"
                + file_uuid
            ),
            "method": "GET",
            "body": None,
            "timeout": 9,
        },
    ]


def test_private_searches_use_separate_vector_service_endpoints(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, *, timeout: float):
        captured.append(
            {
                "url": request.full_url,
                "method": request.method,
                "body": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return FakeResponse({"text": "检索结果"})

    monkeypatch.setattr(client_module, "urlopen", fake_urlopen)
    client = VectorServiceClient(
        VectorServiceSettings(base_url="http://127.0.0.1:8083", timeout=9)
    )

    assert client.search_credential_files(["资质"]) == "检索结果"
    assert client.search_bid_records(["报价"]) == "检索结果"
    assert (
        client.search_credential_files_by_bid_record_id(13, ["证书"])
        == "检索结果"
    )

    assert captured == [
        {
            "url": "http://127.0.0.1:8083/internal/search/credential-files",
            "method": "POST",
            "body": {"queries": ["资质"]},
            "timeout": 9,
        },
        {
            "url": "http://127.0.0.1:8083/internal/search/bid-records",
            "method": "POST",
            "body": {"queries": ["报价"]},
            "timeout": 9,
        },
        {
            "url": (
                "http://127.0.0.1:8083/internal/search/bid-records/13/credentials"
            ),
            "method": "POST",
            "body": {"queries": ["证书"]},
            "timeout": 9,
        },
    ]


def test_health_requires_ready_vector_service(monkeypatch) -> None:
    monkeypatch.setattr(
        client_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse({"status": "not_ready"}),
    )
    client = VectorServiceClient(VectorServiceSettings())

    with pytest.raises(AppException) as exc_info:
        client.health()

    assert exc_info.value.error_code is CHAT_AGENT_UNAVAILABLE
    assert exc_info.value.stage == "validate_vector_service"
