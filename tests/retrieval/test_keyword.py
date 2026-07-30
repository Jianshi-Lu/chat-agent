"""验证最小 Keyword Retriever 的生命周期和基础查询行为。

测试通过 Loader 提供正式的 97 条 ``PolicyDocument``，再调用
``KeywordRetriever.build`` 保存文档集合，最后通过 ``retrieve`` 验证结果、
空查询、Top-K 和未构建异常。测试不读取或过滤 ``interference_ids.txt``。
"""

import pytest

from app.errors import AppException, KNOWLEDGE_RETRIEVAL_FAILED
from app.knowledge.loader import load_documents
from app.retrieval.keyword import KeywordRetriever


@pytest.fixture
def retriever() -> KeywordRetriever:
    """返回已经加载正式本地语料的 Keyword Retriever。"""

    documents = load_documents()
    keyword_retriever = KeywordRetriever()
    keyword_retriever.build(documents)
    return keyword_retriever


def test_build_saves_document_snapshot() -> None:
    """调用方修改原始列表后，已构建的检索器仍保留原文档集合。"""

    documents = load_documents()
    keyword_retriever = KeywordRetriever()
    keyword_retriever.build(documents)
    documents.clear()

    results = keyword_retriever.retrieve("政府采购异常低价如何处理", top_k=3)

    assert results
    assert any(
        result.document_id == "policy-20260114-abnormal-low-price"
        for result in results
    )


def test_retrieve_returns_empty_for_blank_or_unknown_query(
    retriever: KeywordRetriever,
) -> None:
    """空查询和未激活领域词表的查询都直接返回空列表。"""

    assert retriever.retrieve("", top_k=3) == []
    assert retriever.retrieve("今天去哪里旅游", top_k=3) == []


def test_retrieve_respects_top_k(retriever: KeywordRetriever) -> None:
    """查询结果数量不能超过调用方指定的 Top-K。"""

    assert len(retriever.retrieve("政府采购", top_k=1)) <= 1


@pytest.mark.parametrize("query", ["绿色", "政府", "招标", "采购"])
def test_retrieve_supports_short_domain_keywords(
    retriever: KeywordRetriever,
    query: str,
) -> None:
    """查询只包含词表别名的一部分时也应返回候选文档。"""

    assert retriever.retrieve(query, top_k=5)


def test_retrieve_preserves_markdown_content(
    retriever: KeywordRetriever,
) -> None:
    """检索结果中的正文仍然与原始文档正文一致。"""

    documents = load_documents()
    expected_content = next(
        document.content
        for document in documents
        if document.id == "policy-20260114-abnormal-low-price"
    )

    results = retriever.retrieve("异常低价", top_k=10)
    result = next(
        item
        for item in results
        if item.document_id == "policy-20260114-abnormal-low-price"
    )

    assert result.content == expected_content


def test_retrieve_fails_before_build() -> None:
    """未构建文档集合时，检索器抛出统一应用异常。"""

    keyword_retriever = KeywordRetriever()

    with pytest.raises(AppException) as error:
        keyword_retriever.retrieve("政府采购")

    assert error.value.error_code == KNOWLEDGE_RETRIEVAL_FAILED
    assert error.value.stage == "check_lifecycle"


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_rejects_non_positive_top_k(
    retriever: KeywordRetriever,
    top_k: int,
) -> None:
    """Top-K 必须是正数。"""

    with pytest.raises(AppException) as error:
        retriever.retrieve("政府采购", top_k=top_k)

    assert error.value.error_code == KNOWLEDGE_RETRIEVAL_FAILED
    assert error.value.stage == "validate_query"


def run_interactive_keyword_retrieval() -> None:
    """启动一个供本地手工验证使用的 Keyword 检索交互入口。

    函数先加载正式政策文档并构建一次 ``KeywordRetriever``，随后循环读取
    用户输入，将查询交给检索器，并打印命中的文档标题、元数据、分数和
    Markdown 正文。输入 ``q``、``quit`` 或 ``exit`` 退出；直接按回车会
    继续调用检索器并得到空结果。

    该函数不会被 pytest 自动执行，只有直接运行本测试模块时才会启动。
    """

    documents = load_documents()
    keyword_retriever = KeywordRetriever()
    keyword_retriever.build(documents)

    print("Keyword Retriever 已加载 97 条文档。")
    print("请输入关键词；输入 q、quit 或 exit 退出。")

    while True:
        try:
            query = input("\n关键词> ")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return

        if query.strip().casefold() in {"q", "quit", "exit"}:
            print("已退出。")
            return

        results = keyword_retriever.retrieve(query, top_k=5)
        if not results:
            print("未找到匹配文档。")
            continue

        print(f"找到 {len(results)} 条匹配文档：")
        for index, result in enumerate(results, start=1):
            print(f"\n[{index}] {result.title}")
            print(f"ID：{result.document_id}")
            print(f"类型：{result.type}；日期：{result.date}；分数：{result.score}")
            print(f"来源：{result.source}")
            print(f"正文：\n{result.content}")


if __name__ == "__main__":
    run_interactive_keyword_retrieval()
