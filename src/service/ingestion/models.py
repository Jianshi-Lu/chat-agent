"""定义由上游服务提交的稳定知识文件对象。"""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    """表示一个已经转换为 Markdown、等待向量化的知识文件。"""

    document_uuid: str
    content: str
    original_filename: str
    original_object_key: str
    converted_object_key: str
    original_sha256: str
    converted_sha256: str

    def __post_init__(self) -> None:
        """拒绝缺少稳定身份、正文或对象存储定位信息的输入。"""

        try:
            UUID(self.document_uuid)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("document_uuid 必须是有效 UUID") from exc

        required_values = {
            "content": self.content,
            "original_filename": self.original_filename,
            "original_object_key": self.original_object_key,
            "converted_object_key": self.converted_object_key,
            "original_sha256": self.original_sha256,
            "converted_sha256": self.converted_sha256,
        }
        for field_name, value in required_values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} 不能为空")

    def chroma_metadata(self) -> dict[str, str]:
        """返回每个 chunk 都必须携带的对象存储与完整性 metadata。"""

        return {
            "document_uuid": self.document_uuid,
            "original_filename": self.original_filename,
            "original_object_key": self.original_object_key,
            "converted_object_key": self.converted_object_key,
            "original_sha256": self.original_sha256,
            "converted_sha256": self.converted_sha256,
            "format": "md",
            "source": self.converted_object_key,
        }


__all__ = ["KnowledgeDocument"]
