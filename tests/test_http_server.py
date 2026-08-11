"""验证 chat-agent HTTP Server 只暴露对话边界。"""

from fastapi.testclient import TestClient

import src.api.server as http_app


class FakeRuntime:
    def __init__(self) -> None:
        self.questions: list[str] = []
        self.thread_ids: list[str] = []
        self.closed = False

    def answer_with_references(
        self,
        question: str,
        *,
        thread_id: str,
    ):
        self.questions.append(question)
        self.thread_ids.append(thread_id)
        return type(
            "Result",
            (),
            {
                "answer": f"回答：{question}",
                "title": None,
                "referenced_bid_record_ids": (17,),
            },
        )()

    def close(self) -> None:
        self.closed = True


def test_http_lifecycle_and_thread_id(monkeypatch) -> None:
    runtimes: list[FakeRuntime] = []

    def create_runtime() -> FakeRuntime:
        runtime = FakeRuntime()
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(http_app, "Runtime", create_runtime)
    with TestClient(http_app.app) as client:
        assert client.get("/health").json() == {"status": "ready"}
        first = client.post(
            "/chat",
            json={"question": "第一问", "thread_id": "thread-0"},
        )
        second = client.post(
            "/chat",
            json={"question": "第二问", "thread_id": "thread-1"},
        )

    assert first.json() == {
        "answer": "回答：第一问",
        "title": None,
        "referenced_bid_record_ids": [17],
    }
    assert second.json() == {
        "answer": "回答：第二问",
        "title": None,
        "referenced_bid_record_ids": [17],
    }
    assert len(runtimes) == 1
    assert runtimes[0].questions == ["第一问", "第二问"]
    assert runtimes[0].thread_ids == ["thread-0", "thread-1"]
    assert runtimes[0].closed is True


def test_chat_does_not_require_prebound_bid_context(monkeypatch) -> None:
    runtime = FakeRuntime()
    monkeypatch.setattr(http_app, "Runtime", lambda: runtime)

    with TestClient(http_app.app) as client:
        response = client.post(
            "/chat",
            json={
                "question": "任意投标记录的资质是什么？",
                "thread_id": "thread-dynamic",
            },
        )

    assert response.status_code == 200
    assert runtime.thread_ids == ["thread-dynamic"]


def test_missing_thread_id_returns_bad_request(monkeypatch) -> None:
    monkeypatch.setattr(http_app, "Runtime", FakeRuntime)
    with TestClient(http_app.app) as client:
        response = client.post("/chat", json={"question": "问题"})

    assert response.status_code == 400
    assert response.json()["code"] == "KNOWLEDGE_INGESTION_AGENT.CHAT_INPUT_INVALID"


def test_empty_question_returns_bad_request(monkeypatch) -> None:
    monkeypatch.setattr(http_app, "Runtime", FakeRuntime)
    with TestClient(http_app.app) as client:
        response = client.post(
            "/chat",
            json={"question": "", "thread_id": "thread-1"},
        )

    assert response.status_code == 400


def test_vector_endpoints_are_not_exposed_by_chat_agent(monkeypatch) -> None:
    monkeypatch.setattr(http_app, "Runtime", FakeRuntime)
    with TestClient(http_app.app) as client:
        response = client.post(
            "/internal/knowledge-files/vectorize",
            json={"documents": []},
        )

    assert response.status_code == 404
