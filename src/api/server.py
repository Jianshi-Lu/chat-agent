"""chat-agent 的对话 HTTP API 与 Uvicorn 启动入口。"""

from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from configuration import get_http_server_settings
from src.errors import AppException, CHAT_AGENT_UNAVAILABLE, CHAT_INPUT_INVALID
from src.service.runtime import Runtime


class ChatRequest(BaseModel):
    """承载一次带线程标识的对话请求。"""

    question: str = Field(min_length=1)
    thread_id: str = Field(min_length=1, max_length=128)

    @field_validator("thread_id")
    @classmethod
    def normalize_thread_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("thread_id 不能为空")
        return normalized


class ChatResponse(BaseModel):
    """承载一次对话生成的回答；标题仅在首轮生成后返回。"""

    answer: str
    title: str | None = None
    referenced_bid_record_ids: list[int] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """在应用启动时创建 Agent Runtime，在关闭时统一释放。"""

    settings = get_http_server_settings()
    runtime = Runtime()
    app.state.runtime = runtime
    app.state.ready = True
    print(
        f"Chat Agent 已就绪：http://{settings.host}:{settings.port}",
        flush=True,
    )
    try:
        yield
    finally:
        app.state.ready = False
        app.state.runtime = None
        runtime.close()
        print("Chat Agent Runtime 资源已释放。", flush=True)


app = FastAPI(
    title="Chat Agent",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.exception_handler(AppException)
async def handle_app_exception(_: Request, exc: AppException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.error_code.http_status,
        content={
            "code": exc.error_code.code,
            "message": exc.public_message,
            "stage": exc.stage,
        },
    )


@app.exception_handler(RequestValidationError)
async def handle_request_validation(
    _: Request,
    _exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=CHAT_INPUT_INVALID.http_status,
        content={
            "code": CHAT_INPUT_INVALID.code,
            "message": CHAT_INPUT_INVALID.default_message,
            "stage": "validate_request",
        },
    )


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if not getattr(request.app.state, "ready", False):
        return {"status": "not_ready"}
    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    runtime: Runtime | None = getattr(request.app.state, "runtime", None)
    if runtime is None or not getattr(request.app.state, "ready", False):
        raise AppException(
            CHAT_AGENT_UNAVAILABLE,
            module="api.server",
            operation="chat",
            stage="validate_runtime",
            message="知识问答 Runtime 尚未就绪",
        )
    result = runtime.answer_with_references(
        payload.question,
        thread_id=payload.thread_id,
    )
    return ChatResponse(
        answer=result.answer,
        title=result.title,
        referenced_bid_record_ids=list(result.referenced_bid_record_ids),
    )


def _configure_app_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for logger_name in (
        "src.service.runtime",
        "src.service.agent.chat_observer",
    ):
        logging.getLogger(logger_name).setLevel(level)


def main() -> int:
    settings = get_http_server_settings()
    _configure_app_logging(settings.log_level)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "app",
    "chat",
    "health",
    "lifespan",
    "main",
]
