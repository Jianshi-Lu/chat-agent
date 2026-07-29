"""定义知识文档的字段契约和单条数据校验规则。

调用方传入一个包含文档字段的映射，Pydantic 在构造
``PolicyDocument`` 时执行类型转换和字段校验，成功后返回模型实例；失败
时抛出 ``ValidationError``，由上层 Loader 补充文件位置和加载阶段信息。
本模块不读取文件，不判断相关性，不执行集合去重、排序、检索或模型调用。
"""

from datetime import date, datetime
import re

from pydantic import BaseModel, ConfigDict, field_validator


class PolicyDocument(BaseModel):
    """表示一篇已经通过单条字段校验的政策或资讯文档。

    输入：
        包含 ``id``、``title``、``type``、``date``、``source`` 和
        ``content`` 的映射。``id``、``title``、``type``、``source`` 不能为
        空或全空白；``date`` 必须是有效的 ``YYYY-MM-DD``；``content`` 不能
        为空或全空白；未约定字段会被拒绝。

    输出：
        一个字段类型稳定的 Pydantic 模型实例。``date`` 为
        ``datetime.date``，``content`` 保留原始 Markdown 字符串，不截断、不
        去除首尾空白，也不改写 Markdown。

    内部逻辑与异常：
        Pydantic 在实例化时先执行字段类型处理，再执行字段校验器。成功后
        返回模型对象；失败时抛出 ``ValidationError``，由 Loader 转换为
        应用级加载异常。
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
        """校验元数据字符串非空，并返回未改写的原始值。

        参数：
            value: 当前字段经过 Pydantic 初步处理后的值。

        返回：
            原始字符串；``strip`` 只用于判断是否全为空白，不会改变字段值。

        异常：
            值不是字符串、为空或只包含空白字符时抛出 ``ValueError``，由
            Pydantic 汇总为 ``ValidationError``。
        """

        if not isinstance(value, str) or not value.strip():
            raise ValueError("must not be empty or whitespace-only")
        return value

    @field_validator("date", mode="before")
    @classmethod
    def validate_date_format(cls, value: date | str) -> date:
        """在类型转换前校验日期格式，并转换为 ``datetime.date``。

        参数：
            value: 原始日期值，通常来自 JSON 的字符串。

        返回：
            有效的 ``datetime.date``。已经是 ``date`` 的值直接保留。

        异常：
            非 ``YYYY-MM-DD`` 格式或不存在的日期抛出 ``ValueError``；该异常
            最终由 Pydantic 汇总为 ``ValidationError``。
        """

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
        """拒绝空正文，同时原样返回 Markdown 内容。

        参数：
            value: 文档正文字符串。

        返回：
            与输入完全相同的字符串，包括首尾空格、换行和 Markdown 标记。

        异常：
            值不是字符串、为空或只包含空白字符时抛出 ``ValueError``。
        """

        if not isinstance(value, str) or not value.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return value
