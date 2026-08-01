"""已归档的政策分块、向量入库和检索 smoke test。"""

from pathlib import Path

from src.service.ingestion import DEFAULT_POLICIES_PATH, IngestionService
from src.service.retrieval import VectorRetriever


DATA_PATH = DEFAULT_POLICIES_PATH


def main() -> None:
    """加载全部政策文档并打印内存向量库链路结果。"""

    service = IngestionService()
    try:
        chunks = service.ingest(DATA_PATH)
        results = VectorRetriever(service).retrieve(
            "政府采购信息查询相关政策",
            top_k=3,
        )

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
                f"policy_id={metadata['policy_id']} "
                f"text={document[:80]}"
            )
    finally:
        service.close()


if __name__ == "__main__":
    main()
