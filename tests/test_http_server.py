"""验证本地 HTTP Server 的生命周期和请求边界。"""

from fastapi.testclient import TestClient

from configuration import ServerSettings
import src.api.app as http_app


class FakeRuntime:
    """替代真实 Runtime，避免 HTTP 单元测试加载本地模型。"""

    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self.chunks = ["chunk-1", "chunk-2"]
        self.questions: list[str] = []
        self.closed = False

    def ask(self, question: str) -> str:
        self.questions.append(question)
        return f"回答：{question}"

    def close(self) -> None:
        self.closed = True


def test_http_lifecycle_and_stateless_chat(monkeypatch) -> None:
    """请求应复用启动阶段创建的 Runtime，且不传递历史上下文。"""

    settings = ServerSettings()
    runtimes: list[FakeRuntime] = []

    def create_runtime(runtime_settings: ServerSettings) -> FakeRuntime:
        runtime = FakeRuntime(runtime_settings)
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(http_app, "Runtime", create_runtime)
    monkeypatch.setattr(http_app, "get_server_settings", lambda: settings)

    with TestClient(http_app.app) as client:
        assert client.get("/health").json() == {"status": "ready"}
        assert client.post("/chat", json={"question": "第一问"}).json() == {
            "answer": "回答：第一问"
        }
        assert client.post("/chat", json={"question": "第二问"}).json() == {
            "answer": "回答：第二问"
        }

    assert len(runtimes) == 1
    assert runtimes[0].questions == ["第一问", "第二问"]
    assert runtimes[0].closed is True


def test_empty_question_returns_bad_request(monkeypatch) -> None:
    """空问题不能进入 Runtime，应由 HTTP 边界返回 400。"""

    monkeypatch.setattr(http_app, "Runtime", FakeRuntime)
    monkeypatch.setattr(http_app, "get_server_settings", lambda: ServerSettings())

    with TestClient(http_app.app) as client:
        response = client.post("/chat", json={"question": ""})

    assert response.status_code == 400
    assert response.json()["code"] == "KNOWLEDGE_INGESTION_AGENT.CHAT_INPUT_INVALID"
