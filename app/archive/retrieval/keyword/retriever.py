"""实现基于领域概念词表的最小关键词检索器。

调用方先通过 ``build`` 传入完整的 ``PolicyDocument`` 集合；本 MVP 只保存
文档集合，不预先格式化文本或构建倒排索引。调用 ``retrieve`` 时，检索器
临时格式化查询和文档标题、正文，按关键词包含关系和概念组命中情况计算分数，
再返回稳定排序的 ``RetrievedDocument`` 列表。

本模块不读取文件、不读取或过滤 ``interference_ids.txt``，不修改原始文档，
也不调用 LangChain、大模型或向量数据库。文档集合保存在内存中，后续查询
可以复用同一个检索器实例。

当前实现已归档，不属于聊天入口的运行依赖。
"""

from collections.abc import Sequence
from dataclasses import dataclass
import re
import unicodedata

from app.errors import AppException, KNOWLEDGE_RETRIEVAL_FAILED
from app.archive.knowledge.schemas import PolicyDocument
from app.archive.retrieval.schemas import RetrievedDocument

from .terms import KEYWORD_GROUPS


TITLE_WEIGHT = 3.0
CONTENT_WEIGHT = 1.0
EXACT_TITLE_BONUS = 2.0
MIN_SCORE = 1.0
DEFAULT_TOP_K = 3


def _format_for_matching(value: str) -> str:
    """格式化内部匹配文本，不改写调用方传入的原始字符串。

    处理链是 Unicode NFKC 兼容格式化、英文大小写统一以及空白/标点移除。
    返回值只用于本模块的关键词匹配；原始标题和 Markdown 正文仍从
    ``PolicyDocument`` 直接读取并原样返回。
    """

    formatted = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[\W_]+", "", formatted)


@dataclass(frozen=True, slots=True)
class _ScoredDocument:
    """保存一次查询的内部评分结果，不暴露给上层调用方。"""

    document: PolicyDocument
    score: float
    title_hit_count: int


class KeywordRetriever:
    """使用领域概念词表执行最小关键词检索。

    输入：
        ``build`` 接收完整的 ``PolicyDocument`` 序列；``retrieve`` 接收用户
        查询和正数 ``top_k``。

    输出：
        按分数稳定排序的 ``list[RetrievedDocument]``。结果保留完整 Markdown
        正文；无匹配查询返回空列表。

    生命周期：
        ``build`` 只保存文档集合，``retrieve`` 在查询时临时格式化并评分。
        ``_documents is None`` 表示尚未构建，空 tuple 表示已构建但没有文档。
    """

    name = "keyword"

    def __init__(self) -> None:
        """创建尚未加载文档的检索器实例。"""

        self._documents: tuple[PolicyDocument, ...] | None = None

    def build(self, documents: Sequence[PolicyDocument]) -> None:
        """保存本次检索使用的文档集合。

        ``tuple`` 会复制调用方的序列结构，避免调用方后续增删原始列表影响
        检索器。这里不做文件读取、文本格式化、去重、过滤或评分；再次调用
        ``build`` 会用新的文档快照替换旧集合。
        """

        self._documents = tuple(documents)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RetrievedDocument]:
        """根据领域概念命中情况返回最多 ``top_k`` 条文档。

        查询阶段才执行文本格式化，因此当前实现牺牲少量重复计算换取更小的
        内部状态；对于当前 97 条文档的 MVP 数据规模已经足够。
        """

        self._ensure_built()

        if top_k <= 0:
            raise self._error(
                stage="validate_query",
                message="top_k 必须大于 0",
                details={"top_k": top_k},
            )

        formatted_query = _format_for_matching(query)
        if not formatted_query:
            return []

        active_groups = self._active_groups(formatted_query)
        if not active_groups:
            return []

        scored_documents = [
            scored
            for document in self._documents or ()
            if (scored := self._score_document(
                document,
                formatted_query,
                active_groups,
            )) is not None
        ]
        scored_documents.sort(
            key=lambda item: (
                -item.score,
                -item.title_hit_count,
                item.document.id,
            )
        )

        return [
            RetrievedDocument(
                document_id=item.document.id,
                title=item.document.title,
                type=item.document.type,
                date=item.document.date,
                source=item.document.source,
                content=item.document.content,
                score=item.score,
            )
            for item in scored_documents[:top_k]
        ]

    def _ensure_built(self) -> None:
        """确保调用方已经先提供待检索文档。"""

        if self._documents is None:
            raise self._error(
                stage="check_lifecycle",
                message="检索器尚未构建文档集合",
            )

    @staticmethod
    def _active_groups(
        formatted_query: str,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """返回与查询存在包含关系的概念组及匹配词。

        只要“查询包含词表别名”或“词表别名包含查询”，就认为该概念组被
        激活。前者支持用户输入完整问题，后者支持用户只输入别名的一部分，
        例如 ``采购`` 可以命中 ``政府采购``、``采购项目`` 等词。
        返回的匹配词会直接用于文档标题和正文的包含匹配。
        """

        active_groups: list[tuple[str, tuple[str, ...]]] = []
        for group_name, aliases in KEYWORD_GROUPS.items():
            formatted_aliases = tuple(
                _format_for_matching(alias)
                for alias in aliases
            )
            matched_terms: list[str] = []
            for alias in formatted_aliases:
                if alias in formatted_query:
                    matched_terms.append(alias)
                elif formatted_query in alias:
                    matched_terms.append(formatted_query)

            if matched_terms:
                active_groups.append(
                    (group_name, tuple(dict.fromkeys(matched_terms)))
                )
        return tuple(active_groups)

    @staticmethod
    def _score_document(
        document: PolicyDocument,
        formatted_query: str,
        active_groups: tuple[tuple[str, tuple[str, ...]], ...],
    ) -> _ScoredDocument | None:
        """按标题和正文的概念组命中情况计算单篇文档分数。"""

        formatted_title = _format_for_matching(document.title)
        formatted_content = _format_for_matching(document.content)
        score = 0.0
        title_hit_count = 0

        for _, aliases in active_groups:
            title_alias = next(
                (alias for alias in aliases if alias in formatted_title),
                None,
            )
            content_hit = any(
                alias in formatted_content
                for alias in aliases
            )

            if title_alias is not None:
                score += TITLE_WEIGHT
                title_hit_count += 1
                if title_alias in formatted_query:
                    score += EXACT_TITLE_BONUS
            elif content_hit:
                score += CONTENT_WEIGHT

        if score < MIN_SCORE:
            return None

        return _ScoredDocument(
            document=document,
            score=score,
            title_hit_count=title_hit_count,
        )

    @staticmethod
    def _error(
        *,
        stage: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> AppException:
        """构造检索模块统一使用的应用异常。"""

        return AppException(
            KNOWLEDGE_RETRIEVAL_FAILED,
            module="archive.retrieval.keyword",
            operation="retrieve",
            stage=stage,
            message=message,
            details=details,
        )
