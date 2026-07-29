"""知识文档的数据模型。

本模块只负责把输入数据校验并转换为结构化的 ``PolicyDocument``，不负责
读取文件、判断相关性、去重、排序或调用模型。调用链由加载器或上层服务
传入一个文档对象，Pydantic 完成字段和类型校验后返回模型实例；校验失败
时抛出 ``ValidationError``，由上层加载流程统一转换为加载异常。
"""

from datetime import date, datetime
import re

from pydantic import BaseModel, ConfigDict, field_validator


class PolicyDocument(BaseModel):
    """知识库中的一篇政策或资讯文档。

    输入是包含 ``id``、``title``、``type``、``date``、``source`` 和
    ``content`` 的映射；输出是字段类型稳定的 Pydantic 模型实例。其中
    ``date`` 会转换为 ``datetime.date``，Markdown 正文 ``content`` 保持
    原始字符串不变，供后续检索器直接消费。``extra="forbid"`` 防止数据
    文件悄悄混入未约定字段，避免后续处理链对字段含义产生歧义。

    Pydantic 是本项目使用的数据校验依赖：模型声明定义数据契约，字段校验
    器在实例化时执行，成功后返回模型对象，失败时抛出结构化的
    ``ValidationError``。加载器应负责捕获并补充文档位置等上下文。
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    type: str
    date: date
    source: str
    content: str

    @field_validator("id", "title", "type", "source")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        """校验文档元数据非空，并返回原值而不改写调用方输入。"""

        if not isinstance(value, str) or not value.strip():
            raise ValueError("must not be empty or whitespace-only")
        return value

    @field_validator("date", mode="before")
    @classmethod
    def validate_date_format(cls, value: date | str) -> date:
        """在 Pydantic 类型转换前，要求日期使用有效的 ``YYYY-MM-DD``。"""

        if isinstance(value, datetime):
            raise ValueError("date must use YYYY-MM-DD")

        if isinstance(value, date):
            return value

        if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
            raise ValueError("date must use YYYY-MM-DD")

        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("date must use a valid YYYY-MM-DD value") from exc

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """拒绝空正文，同时原样保留 Markdown 的空格、换行和标记。"""

        if not isinstance(value, str) or not value.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return value
