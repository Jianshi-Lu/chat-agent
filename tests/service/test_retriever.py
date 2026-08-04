"""验证官方 VectorStoreRetriever 到 Agent Tool 的最小边界。"""

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from src.service.agent.tools import build_retriever_tool


def _build_retriever():
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=3))
    vector_store.add_documents(
        [
            Document(
                page_content="政府采购公告正文",
                metadata={
                    "id": "policy-1-0",
                    "source": "财政部",
                },
            ),
            Document(
                page_content="政府采购信息发布要求",
                metadata={
                    "id": "policy-2-0",
                    "source": "财政部",
                },
            ),
        ],
        ids=["policy-1-0", "policy-2-0"],
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )


def test_official_retriever_returns_langchain_documents() -> None:
    retriever = _build_retriever()

    result = retriever.invoke("政府采购公告是什么？")

    assert len(result) == 2
    assert all(isinstance(document, Document) for document in result)
    assert {document.metadata["id"] for document in result} == {
        "policy-1-0",
        "policy-2-0",
    }


def test_build_retriever_tool_returns_formatted_documents() -> None:
    tool = build_retriever_tool(_build_retriever())

    result = tool.invoke({"query": "政府采购公告是什么？"})

    assert tool.name == "retrieve"
    assert "文档ID：" in result
    assert "来源：" in result
    assert "正文：" in result
