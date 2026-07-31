"""已归档的分块、写入内存 Chroma 和查询 smoke test。"""

from pathlib import Path

from app.archive.knowledge.loader import load_documents
from app.service.ingestion import IngestionService


DATA_PATH = Path(
    "/Users/lujianshi/Desktop/projects/knowledge-ingestion-agent/data/policies.json"
)


def main() -> None:
    """加载全部政策文档并打印内存向量库链路结果。"""

    documents = load_documents(DATA_PATH)
    service = IngestionService()
    chunks = service.ingest(documents)
    results = service.search("政府采购信息查询相关政策", top_k=3)

    print(f"原始文档数量：{len(documents)}")
    print(f"分块数量：{len(chunks)}")
    print(f"Chroma collection 数量：{service.collection.count()}")
    print("查询结果：")
    for rank, (document, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print(
            f"{rank}. distance={distance:.6f} "
            f"document_id={metadata['document_id']} "
            f"text={document[:80]}"
        )


if __name__ == "__main__":
    main()
