"""Loader 的最小手工观察测试。

测试输入是 Loader 默认读取的 ``data/policies.json``，内部调用
``load_documents`` 完成文件读取和文档模型构造，输出是打印到终端的全部
文档内容。当前测试不做额外业务判断，主要用于直观看到加载结果。
"""

from pprint import pprint

from app.knowledge.loader import load_documents


def test_print_all_documents() -> None:
    """加载并打印当前全部政策文档。

    输入：Loader 默认数据文件。
    内部逻辑：调用 ``load_documents``，再将每个 ``PolicyDocument`` 转成可读
    的 JSON 兼容字典，避免直接打印模型对象时隐藏字段结构。
    输出：测试运行终端中的完整文档列表；Loader 抛出异常时测试失败。
    """

    documents = load_documents()
    pprint(
        [document.model_dump(mode="json") for document in documents],
        sort_dicts=False,
    )

test_print_all_documents()