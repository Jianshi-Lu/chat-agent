"""按文件格式解析源资料，不约定业务字段结构。"""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.errors import AppException, KNOWLEDGE_DOCUMENT_LOAD_FAILED


def parse_documents(path: str | Path) -> list[Document]:
    """解析 Markdown 或 JSON，返回尚未进行 Markdown 切分的 Document。"""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return [_parse_markdown(file_path)]
    if suffix == ".json":
        return _parse_json(file_path)
    raise _error(
        stage="validate_format",
        message="暂只支持 Markdown（.md/.markdown）和 JSON（.json）文件",
        details={"file_path": str(file_path)},
    )


def _parse_markdown(path: Path) -> Document:
    """将整个 Markdown 文件读取为一个待定长切分的 Document。"""

    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _error(
            stage="read_markdown",
            message="Markdown 文件读取失败",
            details={"file_path": str(path)},
        ) from exc

    return Document(
        page_content=content,
        metadata={
            "source": str(path),
            "format": "md",
            "document_index": 0,
        },
    )


def _parse_json(path: Path) -> list[Document]:
    """将 JSON 顶层 object 或 object 数组转换为独立 Document。"""

    payload = _load_json(path)
    if isinstance(payload, dict):
        objects = [payload]
    elif isinstance(payload, list):
        objects = payload
    else:
        raise _error(
            stage="validate_json",
            message="JSON 顶层结构必须是 object 或 object 数组",
            details={"file_path": str(path)},
        )

    documents: list[Document] = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise _error(
                stage="validate_json",
                message="JSON 数组中的每个元素必须是 object",
                details={"file_path": str(path), "document_index": index},
            )
        documents.append(
            Document(
                page_content=json.dumps(
                    item,
                    ensure_ascii=False,
                    indent=2,
                ),
                metadata={
                    "source": str(path),
                    "format": "json",
                    "document_index": index,
                },
            )
        )
    return documents


def _load_json(path: Path) -> Any:
    """读取 JSON 原始数据，统一转换文件读取和解析错误。"""

    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise _error(
            stage="parse_json",
            message=f"JSON 格式错误：第 {exc.lineno} 行第 {exc.colno} 列",
            details={"file_path": str(path)},
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise _error(
            stage="read_json",
            message="JSON 文件读取失败",
            details={"file_path": str(path)},
        ) from exc


def _error(
    *,
    stage: str,
    message: str,
    details: dict[str, object] | None = None,
) -> AppException:
    """构造 parser 模块的结构化异常。"""

    return AppException(
        KNOWLEDGE_DOCUMENT_LOAD_FAILED,
        module="service.ingestion.parser",
        operation="parse_documents",
        stage=stage,
        message=message,
        details=details,
    )


__all__ = ["parse_documents"]
