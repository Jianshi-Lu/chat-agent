"""将本地 JSON 文件转换为经过校验的知识文档对象。

本模块位于文件系统和文档模型之间。调用方通过 ``load_documents`` 传入路径；
模块依次检查路径、读取 UTF-8 文本、解析 JSON、构造 ``PolicyDocument``，再做
全局 ID 唯一性校验，最后返回 ``list[PolicyDocument]``。读取或校验失败时，
底层异常会被包装为带错误代码和处理阶段的 ``AppException``。

当前 MVP 一次性把文件和模型列表加载到内存中。模块只负责读取、结构校验和
集合校验，不负责相关性判断、去重清洗、排序、检索、分块、向量化或调用模型；
因此干扰数据会原样进入返回结果，交由后续处理阶段决定如何使用。

当前实现已归档，不属于聊天入口的运行依赖。
"""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.errors import (
    AppException,
    KNOWLEDGE_DOCUMENT_LOAD_FAILED,
)

from .schemas import PolicyDocument


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICIES_PATH = PROJECT_ROOT / "data" / "policies.json"


def _parse_documents(payload: Any) -> list[PolicyDocument]:
    """把已解析的 JSON 载荷转换为文档模型列表。

    参数：
        payload: ``json.loads`` 返回的任意 Python 对象，预期是由文档对象组成
            的列表。

    返回：
        按原 JSON 顺序排列的 ``list[PolicyDocument]``。Pydantic 会在构造时
        执行字段类型和字段值校验。

    逻辑：
        先确认顶层是列表，再逐条确认元素是字典并调用
        ``PolicyDocument.model_validate``。校验失败时补充文档序号后抛出
        ``AppException``；不会返回部分成功结果。
    """

    if not isinstance(payload, list):
        raise AppException(
            KNOWLEDGE_DOCUMENT_LOAD_FAILED,
            module="archive.knowledge.loader",
            operation="load_documents",
            stage="validate_structure",
            message="政策文档 JSON 顶层结构必须是数组",
        )

    documents: list[PolicyDocument] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise AppException(
                KNOWLEDGE_DOCUMENT_LOAD_FAILED,
                module="archive.knowledge.loader",
                operation="load_documents",
                stage="validate_document",
                message=f"第 {index} 条文档校验失败：必须是 JSON 对象",
                details={"document_index": index},
            )

        try:
            documents.append(PolicyDocument.model_validate(item))
        except ValidationError as exc:
            reason = str(exc)
            raise AppException(
                KNOWLEDGE_DOCUMENT_LOAD_FAILED,
                module="archive.knowledge.loader",
                operation="load_documents",
                stage="validate_document",
                message=f"第 {index} 条文档校验失败：{reason}",
                details={"document_index": index, "validation_message": reason},
            ) from exc

    return documents


def _ensure_unique_ids(documents: list[PolicyDocument]) -> None:
    """检查文档集合中的 ID 是否全局唯一。

    参数：
        documents: 已通过单条字段校验的文档列表。

    返回：
        校验通过时无返回值。

    逻辑与异常：
        使用集合记录已出现的 ID；再次出现同一 ID 时抛出带重复 ID 的
        ``AppException``。该方法只检查唯一性，不删除或合并重复文档。
    """

    seen_ids: set[str] = set()
    for document in documents:
        if document.id in seen_ids:
            raise AppException(
                KNOWLEDGE_DOCUMENT_LOAD_FAILED,
                module="archive.knowledge.loader",
                operation="load_documents",
                stage="validate_collection",
                message=f"发现重复文档 ID：{document.id}",
                details={"document_id": document.id},
            )
        seen_ids.add(document.id)


def load_documents(
    path: str | Path = DEFAULT_POLICIES_PATH,
) -> list[PolicyDocument]:
    """读取文件并返回完整的政策文档集合。

    参数：
        path: JSON 数据文件路径，可以是字符串或 ``Path``。省略时使用项目
            根目录下的 ``data/policies.json``。

    返回：
        所有校验通过的 ``PolicyDocument`` 列表，顺序与 JSON 数组一致；其中
        ``date`` 已转换为 ``datetime.date``，正文仍保持原始 Markdown。

    内部逻辑：
        先准备并检查文件路径，再以 UTF-8 读取文本并解析 JSON；随后调用
        ``_parse_documents`` 做顶层结构和单条文档校验，调用
        ``_ensure_unique_ids`` 做集合级校验。

    异常边界：
        任一阶段失败都抛出 ``AppException``，不会返回部分结果，也不会读取
        或过滤 ``interference_ids.txt`` 中的干扰文档。
    """

    try:
        file_path = Path(path)
    except TypeError as exc:
        raise AppException(
            KNOWLEDGE_DOCUMENT_LOAD_FAILED,
            module="archive.knowledge.loader",
            operation="load_documents",
            stage="prepare_path",
            message=f"政策文档路径无效：{path!r}",
        ) from exc

    try:
        if not file_path.exists():
            raise AppException(
                KNOWLEDGE_DOCUMENT_LOAD_FAILED,
                module="archive.knowledge.loader",
                operation="load_documents",
                stage="check_file",
                message=f"政策文档文件不存在：{file_path}",
                details={"reason": "file_not_found", "path": str(file_path)},
            )
        if not file_path.is_file():
            raise AppException(
                KNOWLEDGE_DOCUMENT_LOAD_FAILED,
                module="archive.knowledge.loader",
                operation="load_documents",
                stage="check_file",
                message=f"政策文档路径不是文件：{file_path}",
                details={"reason": "path_not_file", "path": str(file_path)},
            )
    except OSError as exc:
        raise AppException(
            KNOWLEDGE_DOCUMENT_LOAD_FAILED,
                module="archive.knowledge.loader",
            operation="load_documents",
            stage="check_file",
            message="政策文档文件检查失败",
            details={"reason": "file_check_failed"},
        ) from exc

    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AppException(
            KNOWLEDGE_DOCUMENT_LOAD_FAILED,
            module="archive.knowledge.loader",
            operation="load_documents",
            stage="read_file",
            message="政策文档文件读取失败",
            details={"reason": "file_read_failed", "path": str(file_path)},
        ) from exc

    try:
        payload = json.loads(text)
    except JSONDecodeError as exc:
        raise AppException(
            KNOWLEDGE_DOCUMENT_LOAD_FAILED,
            module="archive.knowledge.loader",
            operation="load_documents",
            stage="parse_json",
            message=f"政策文档 JSON 格式错误：第 {exc.lineno} 行第 {exc.colno} 列",
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc

    documents = _parse_documents(payload)
    _ensure_unique_ids(documents)
    return documents
