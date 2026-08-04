"""验证项目 Retriever 到 LangChain retriever tool 的最小边界。"""

from src.service.agent.tools import build_retriever_tool
from src.service.retrieval import Retriever


class _FakeEmbedding:
    def encode(self, query: str, **kwargs):
        assert query == "政府采购公告是什么？"
        assert kwargs["prompt_name"] == "query"
        return self

    def tolist(self):
        return [0.1, 0.2, 0.3]


class _FakeCollection:
    def count(self) -> int:
        return 2

    def query(self, **kwargs):
        assert kwargs["n_results"] == 2
        return {
            "ids": [["policy-1-0", "policy-2-0"]],
            "documents": [["政府采购公告正文", "政府采购信息发布要求"]],
            "metadatas": [
                [
                    {
                        "title": "政府采购信息发布管理办法",
                        "date": "2019-11-27",
                        "source": "财政部",
                    },
                    {
                        "title": "政府采购信息公开通知",
                        "date": "2020-01-01",
                        "source": "财政部",
                    },
                ]
            ],
            "distances": [[0.1, 0.2]],
        }


class _FakeIngestion:
    embedding_model = _FakeEmbedding()
    policy_chunk_collection = _FakeCollection()


def test_retriever_returns_raw_chroma_results() -> None:
    policy_retriever = Retriever(_FakeIngestion())

    result = policy_retriever.retrieve("政府采购公告是什么？", top_k=2)

    assert result["documents"] == [
        [
            "政府采购公告正文",
            "政府采购信息发布要求",
        ]
    ]
    assert result["ids"] == [["policy-1-0", "policy-2-0"]]
    assert result["metadatas"][0][0]["title"] == "政府采购信息发布管理办法"
    assert result["distances"] == [[0.1, 0.2]]


def test_build_retriever_tool_returns_formatted_documents() -> None:
    policy_retriever = Retriever(_FakeIngestion())
    tool = build_retriever_tool(policy_retriever, top_k=2)

    result = tool.invoke({"query": "政府采购公告是什么？"})

    assert tool.name == "retrieve"
    assert "文档ID：policy-1-0" in result
    assert "标题：政府采购信息发布管理办法" in result
    assert "正文：\n政府采购公告正文" in result
