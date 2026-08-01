"""定义本地知识问答 HTTP API 和 FastAPI 生命周期。

本模块只负责 HTTP 边界：请求模型、路由、应用异常转换和 Runtime 生命周期。
模型权重、持久化 Chroma、检索器和 LangChain Agent 由
``src.service.runtime.Runtime`` 持有，
不会在每个请求中重复初始化。
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from configuration import get_http_server_settings
from src.errors import AppException, CHAT_AGENT_UNAVAILABLE, CHAT_INPUT_INVALID
from src.service.runtime import Runtime


class ChatRequest(BaseModel):
    """承载一次无状态对话请求。"""

    question: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """承载一次对话生成的回答。"""

    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """在应用启动时创建 Runtime，在应用关闭时释放全部资源。"""

    settings = get_http_server_settings()
    runtime = Runtime()
    app.state.runtime = runtime
    app.state.ready = True
    print(
        f"知识库初始化完成：{runtime.chunk_count} 个文本块\n"
        f"HTTP Server 已启动：http://{settings.host}:{settings.port}",
        flush=True,
    )

    try:
        yield
    finally:
        app.state.ready = False
        app.state.runtime = None
        runtime.close()
        print("Runtime 资源已释放。", flush=True)


app = FastAPI(
    title="Knowledge Ingestion Agent",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.exception_handler(AppException)
async def handle_app_exception(_: Request, exc: AppException) -> JSONResponse:
    """将应用异常转换为不暴露底层堆栈的统一 JSON 响应。"""

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
    """将请求体校验错误统一转换为 400 响应。"""

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
    """返回当前 HTTP Server 是否完成 Runtime 初始化。"""

    if not getattr(request.app.state, "ready", False):
        return {"status": "not_ready"}
    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """使用当前问题执行一次独立的检索增强问答。"""

    runtime: Runtime | None = getattr(request.app.state, "runtime", None)
    if runtime is None or not getattr(request.app.state, "ready", False):
        raise AppException(
            CHAT_AGENT_UNAVAILABLE,
            module="api.app",
            operation="chat",
            stage="validate_runtime",
            message="知识问答 Runtime 尚未就绪",
        )

    return ChatResponse(answer=runtime.answer(payload.question))


__all__ = ["ChatRequest", "ChatResponse", "app", "lifespan"]
