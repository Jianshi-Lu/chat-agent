"""定义并启动本地知识问答 HTTP API。

本模块同时负责 HTTP 边界和 Uvicorn 启动入口：请求模型、路由、应用异常转换、
Runtime 生命周期以及 HTTP Server 配置都集中在这里。模型权重、持久化 Chroma、
检索器和 LangChain Agent 由 ``src.service.runtime.Runtime`` 持有，不会在每个请求中
重复初始化。
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import UUID

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from configuration import get_http_server_settings
from src.errors import AppException, CHAT_AGENT_UNAVAILABLE, CHAT_INPUT_INVALID
from src.service.ingestion import KnowledgeDocument
from src.service.runtime import Runtime


class ChatRequest(BaseModel):
    """承载一次带线程标识的对话请求。"""

    question: str = Field(min_length=1)
    thread_id: str = Field(min_length=1, max_length=128)

    @field_validator("thread_id")
    @classmethod
    def normalize_thread_id(cls, value: str) -> str:
        """拒绝空白线程标识，并统一去除首尾空格。"""

        normalized = value.strip()
        if not normalized:
            raise ValueError("thread_id 不能为空")
        return normalized


class ChatResponse(BaseModel):
    """承载一次对话生成的回答；``title`` 仅在首轮自动生成后返回。"""

    answer: str
    title: str | None = None


class VectorizeDocumentRequest(BaseModel):
    """承载一个 converted Markdown 及其稳定对象存储 metadata。"""

    model_config = ConfigDict(populate_by_name=True)

    document_uuid: UUID = Field(alias="documentUuid")
    content: str = Field(min_length=1)
    original_filename: str = Field(alias="originalFilename", min_length=1)
    original_object_key: str = Field(alias="originalObjectKey", min_length=1)
    converted_object_key: str = Field(alias="convertedObjectKey", min_length=1)
    original_sha256: str = Field(
        alias="originalSha256",
        pattern=r"^[0-9a-f]{64}$",
    )
    converted_sha256: str = Field(
        alias="convertedSha256",
        pattern=r"^[0-9a-f]{64}$",
    )

    def to_domain(self) -> KnowledgeDocument:
        """转换为不依赖 FastAPI/Pydantic 的 ingestion 输入。"""

        return KnowledgeDocument(
            document_uuid=str(self.document_uuid),
            content=self.content,
            original_filename=self.original_filename,
            original_object_key=self.original_object_key,
            converted_object_key=self.converted_object_key,
            original_sha256=self.original_sha256,
            converted_sha256=self.converted_sha256,
        )


class VectorizeDocumentsRequest(BaseModel):
    """承载一次 Java 请求中的全部知识文件。"""

    documents: list[VectorizeDocumentRequest] = Field(min_length=1)


class VectorizedDocumentResponse(BaseModel):
    """返回单个知识文件实际写入的 chunk 数量。"""

    model_config = ConfigDict(populate_by_name=True)

    document_uuid: UUID = Field(alias="documentUuid")
    chunk_count: int = Field(alias="chunkCount", ge=1)


class VectorizeDocumentsResponse(BaseModel):
    """返回一次批量向量化的全部结果。"""

    documents: list[VectorizedDocumentResponse]


class DeleteKnowledgeDocumentResponse(BaseModel):
    """返回按 document_uuid 删除的 chunk 数量。"""

    model_config = ConfigDict(populate_by_name=True)

    document_uuid: UUID = Field(alias="documentUuid")
    deleted_chunk_count: int = Field(alias="deletedChunkCount", ge=0)


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
    """按 thread_id 执行一次可恢复上下文的检索增强问答。"""

    runtime: Runtime | None = getattr(request.app.state, "runtime", None)
    if runtime is None or not getattr(request.app.state, "ready", False):
        raise AppException(
            CHAT_AGENT_UNAVAILABLE,
            module="api.server",
            operation="chat",
            stage="validate_runtime",
            message="知识问答 Runtime 尚未就绪",
        )

    answer, title = runtime.answer_with_title(
        payload.question,
        thread_id=payload.thread_id,
    )
    return ChatResponse(answer=answer, title=title)


@app.post(
    "/internal/knowledge-files/vectorize",
    response_model=VectorizeDocumentsResponse,
)
def vectorize_knowledge_files(
    request: Request,
    payload: VectorizeDocumentsRequest,
) -> VectorizeDocumentsResponse:
    """向量化 Java 已转换并持久化的一个或多个 Markdown 文件。"""

    runtime: Runtime | None = getattr(request.app.state, "runtime", None)
    if runtime is None or not getattr(request.app.state, "ready", False):
        raise AppException(
            CHAT_AGENT_UNAVAILABLE,
            module="api.server",
            operation="vectorize_knowledge_files",
            stage="validate_runtime",
            message="知识向量化 Runtime 尚未就绪",
        )

    results = runtime.vectorize_documents(
        [document.to_domain() for document in payload.documents]
    )
    return VectorizeDocumentsResponse(
        documents=[
            VectorizedDocumentResponse(
                documentUuid=document_uuid,
                chunkCount=chunk_count,
            )
            for document_uuid, chunk_count in results
        ]
    )


@app.delete(
    "/internal/knowledge-files/{document_uuid}",
    response_model=DeleteKnowledgeDocumentResponse,
)
def delete_knowledge_file(
    request: Request,
    document_uuid: UUID,
) -> DeleteKnowledgeDocumentResponse:
    """按 document_uuid 幂等删除一个知识文件的全部向量。"""

    runtime: Runtime | None = getattr(request.app.state, "runtime", None)
    if runtime is None or not getattr(request.app.state, "ready", False):
        raise AppException(
            CHAT_AGENT_UNAVAILABLE,
            module="api.server",
            operation="delete_knowledge_file",
            stage="validate_runtime",
            message="知识向量化 Runtime 尚未就绪",
        )

    deleted_count = runtime.delete_document(str(document_uuid))
    return DeleteKnowledgeDocumentResponse(
        documentUuid=document_uuid,
        deletedChunkCount=deleted_count,
    )


def main() -> int:
    """读取配置并启动单 worker Uvicorn Server。"""

    settings = get_http_server_settings()
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
    "DeleteKnowledgeDocumentResponse",
    "VectorizeDocumentRequest",
    "VectorizeDocumentsRequest",
    "VectorizeDocumentsResponse",
    "VectorizedDocumentResponse",
    "app",
    "chat",
    "delete_knowledge_file",
    "health",
    "lifespan",
    "main",
    "vectorize_knowledge_files",
]
