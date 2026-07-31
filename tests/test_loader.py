"""验证归档 Loader 可以加载默认政策文档。

测试输入是 Loader 默认读取的 ``data/policies.json``，内部调用
``load_documents`` 完成文件读取和文档模型构造，并验证当前数据数量和 ID
唯一性。归档 Loader 不属于聊天入口的运行依赖。
"""

from app.archive.knowledge.loader import load_documents


def test_load_default_documents() -> None:
    """加载默认政策文档，并验证数量和全局 ID 唯一性。"""

    documents = load_documents()

    assert len(documents) == 97
    assert len({document.id for document in documents}) == 97
