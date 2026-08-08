"""验证本地 HTTP Server 的生命周期和请求边界。"""

from fastapi.testclient import TestClient

import src.api.server as http_app
from src.service.runtime import VectorizeOutcome


class FakeRuntime:
    """替代真实 Runtime，避免 HTTP 单元测试加载本地模型。"""

    def __init__(self) -> None:
        self.chunks = ["chunk-1", "chunk-2"]
        self.chunk_count = len(self.chunks)
        self.questions: list[str] = []
        self.thread_ids: list[str] = []
        self.closed = False
        self.vectorized_document_uuids: list[str] = []
        self.deleted_document_uuids: list[str] = []

    def answer(
        self,
        question: str,
        *,
        thread_id: str,
    ) -> str:
        self.questions.append(question)
        self.thread_ids.append(thread_id)
        return f"回答：{question}"

    def answer_with_title(
        self,
        question: str,
        *,
        thread_id: str,
    ) -> tuple[str, str | None]:
        return self.answer(question, thread_id=thread_id), None

    def close(self) -> None:
        self.closed = True

    def vectorize_documents(self, documents):
        self.vectorized_document_uuids.extend(
            document.document_uuid for document in documents
        )
        return [
            VectorizeOutcome(document.document_uuid, 2)
            for document in documents
        ]

    def get_document_chunk_count(self, document_uuid: str) -> int:
        return 2

    def delete_document(self, document_uuid: str) -> int:
        self.deleted_document_uuids.append(document_uuid)
        return 2


def test_http_lifecycle_and_thread_id(monkeypatch) -> None:
    """请求应复用启动阶段创建的 Runtime，并接受线程标识。"""

    runtimes: list[FakeRuntime] = []

    def create_runtime() -> FakeRuntime:
        runtime = FakeRuntime()
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(http_app, "Runtime", create_runtime)

    with TestClient(http_app.app) as client:
        assert client.get("/health").json() == {"status": "ready"}
        assert client.post(
            "/chat",
            json={"question": "第一问", "thread_id": "thread-0"},
        ).json() == {
            "answer": "回答：第一问",
            "title": None,
        }
        assert client.post(
            "/chat",
            json={"question": "第二问", "thread_id": "thread-1"},
        ).json() == {
            "answer": "回答：第二问",
            "title": None,
        }

    assert len(runtimes) == 1
    assert runtimes[0].questions == ["第一问", "第二问"]
    assert runtimes[0].thread_ids == ["thread-0", "thread-1"]
    assert runtimes[0].closed is True


def test_missing_thread_id_returns_bad_request(monkeypatch) -> None:
    """缺少 thread_id 时不能回退到 default。"""

    monkeypatch.setattr(http_app, "Runtime", FakeRuntime)

    with TestClient(http_app.app) as client:
        response = client.post("/chat", json={"question": "问题"})

    assert response.status_code == 400
    assert response.json()["code"] == "KNOWLEDGE_INGESTION_AGENT.CHAT_INPUT_INVALID"


def test_empty_question_returns_bad_request(monkeypatch) -> None:
    """空问题不能进入 Runtime，应由 HTTP 边界返回 400。"""

    monkeypatch.setattr(http_app, "Runtime", FakeRuntime)

    with TestClient(http_app.app) as client:
        response = client.post("/chat", json={"question": ""})

    assert response.status_code == 400
    assert response.json()["code"] == "KNOWLEDGE_INGESTION_AGENT.CHAT_INPUT_INVALID"


def test_vectorize_and_delete_knowledge_document(monkeypatch) -> None:
    """HTTP 接口应把稳定知识文件交给 Runtime，并按 UUID 幂等删除。"""

    runtimes: list[FakeRuntime] = []

    def create_runtime() -> FakeRuntime:
        runtime = FakeRuntime()
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(http_app, "Runtime", create_runtime)
    document_uuid = "550e8400-e29b-41d4-a716-446655440000"
    payload = {
        "documents": [
            {
                "documentUuid": document_uuid,
                "content": "# 测试知识\n\n唯一事实。",
                "originalFilename": "knowledge.txt",
                "originalObjectKey": f"{document_uuid}/original/knowledge.txt",
                "convertedObjectKey": f"{document_uuid}/converted/content.md",
                "originalSha256": "a" * 64,
                "convertedSha256": "b" * 64,
            }
        ]
    }

    with TestClient(http_app.app) as client:
        vectorize_response = client.post(
            "/internal/knowledge-files/vectorize",
            json=payload,
        )
        delete_response = client.delete(
            f"/internal/knowledge-files/{document_uuid}"
        )

    assert vectorize_response.status_code == 200
    assert vectorize_response.json() == {
        "documents": [
            {
                "documentUuid": document_uuid,
                "success": True,
                "chunkCount": 2,
                "message": None,
            }
        ]
    }
    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "documentUuid": document_uuid,
        "deletedChunkCount": 2,
    }
    assert runtimes[0].vectorized_document_uuids == [document_uuid]
    assert runtimes[0].deleted_document_uuids == [document_uuid]


def test_vector_status_returns_existing_chunk_count(monkeypatch) -> None:
    """状态查询应返回当前 Chroma 中的 chunk 数，供 Java 对账与重试。"""

    runtimes: list[FakeRuntime] = []

    def create_runtime() -> FakeRuntime:
        runtime = FakeRuntime()
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(http_app, "Runtime", create_runtime)
    document_uuid = "550e8400-e29b-41d4-a716-446655440000"

    with TestClient(http_app.app) as client:
        response = client.get(
            f"/internal/knowledge-files/{document_uuid}"
        )

    assert response.status_code == 200
    assert response.json() == {
        "documentUuid": document_uuid,
        "chunkCount": 2,
    }


def test_vectorize_returns_per_document_failure(monkeypatch) -> None:
    """单个文档向量化失败应作为该文档结果返回，不连坐整批。"""

    class PartiallyFailingRuntime(FakeRuntime):
        def vectorize_documents(self, documents):
            self.vectorized_document_uuids.extend(
                document.document_uuid for document in documents
            )
            return [
                VectorizeOutcome(document.document_uuid, 2)
                if index == 0
                else VectorizeOutcome(
                    document.document_uuid,
                    error_message="切分失败",
                )
                for index, document in enumerate(documents)
            ]

    monkeypatch.setattr(http_app, "Runtime", PartiallyFailingRuntime)
    first_uuid = "550e8400-e29b-41d4-a716-446655440000"
    second_uuid = "660e8400-e29b-41d4-a716-446655440000"
    payload = {
        "documents": [
            {
                "documentUuid": first_uuid,
                "content": "# 第一份",
                "originalFilename": "a.txt",
                "originalObjectKey": f"{first_uuid}/original/a.txt",
                "convertedObjectKey": f"{first_uuid}/converted/content.md",
                "originalSha256": "a" * 64,
                "convertedSha256": "b" * 64,
            },
            {
                "documentUuid": second_uuid,
                "content": "# 第二份",
                "originalFilename": "b.txt",
                "originalObjectKey": f"{second_uuid}/original/b.txt",
                "convertedObjectKey": f"{second_uuid}/converted/content.md",
                "originalSha256": "c" * 64,
                "convertedSha256": "d" * 64,
            },
        ]
    }

    with TestClient(http_app.app) as client:
        response = client.post(
            "/internal/knowledge-files/vectorize",
            json=payload,
        )

    assert response.status_code == 200
    assert response.json() == {
        "documents": [
            {
                "documentUuid": first_uuid,
                "success": True,
                "chunkCount": 2,
                "message": None,
            },
            {
                "documentUuid": second_uuid,
                "success": False,
                "chunkCount": None,
                "message": "切分失败",
            },
        ]
    }
