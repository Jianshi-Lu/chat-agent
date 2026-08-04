"""已归档的政策分块、向量入库和检索 smoke test。"""

from configuration import get_ingestion_settings, get_retrieval_settings
from src.service.ingestion import IngestionPipeline


INGESTION_SETTINGS = get_ingestion_settings()
RETRIEVAL_SETTINGS = get_retrieval_settings()


def main() -> None:
    """加载全部政策文档并打印内存向量库链路结果。"""

    pipeline = IngestionPipeline(
        model_path=INGESTION_SETTINGS.embedding_model_path,
        chroma_path=INGESTION_SETTINGS.chroma_path,
        collection_name=INGESTION_SETTINGS.collection_name,
        embedding_batch_size=INGESTION_SETTINGS.embedding_batch_size,
        query_embedding_batch_size=INGESTION_SETTINGS.query_embedding_batch_size,
    )
    try:
        chunks = pipeline.run(
            INGESTION_SETTINGS.source_path,
            chunk_size=INGESTION_SETTINGS.chunk_size,
            chunk_overlap=INGESTION_SETTINGS.chunk_overlap,
        )
        retriever = pipeline.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_SETTINGS.top_k},
        )
        results = retriever.invoke("政府采购信息查询相关政策")

        print(f"分块数量：{len(chunks)}")
        collection_count = len(pipeline.vector_store.get(include=[])["ids"])
        print(f"Chroma collection 数量：{collection_count}")
        print("查询结果：")
        for rank, document in enumerate(results, start=1):
            print(
                f"{rank}. source={document.metadata['source']} "
                f"text={document.page_content[:80]}"
            )
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
