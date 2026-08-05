"""验证 ingestion 各职责模块和 Pipeline 组装边界。"""

import json
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import src.service.ingestion.pipeline as pipeline_module
from src.service.ingestion.parser import parse_documents
from src.service.ingestion.splitter import create_splitter
from src.service.ingestion.pipeline import IngestionPipeline
from src.service.ingestion.models import KnowledgeDocument


def _policy_payload() -> list[dict[str, str]]:
    return [
        {
            "name": "第一条数据",
            "content": "JSON 对象正文。",
        }
    ]


def test_json_parser_uses_each_object_as_document_unit(tmp_path) -> None:
    path = tmp_path / "policies.json"
    path.write_text(
        json.dumps(_policy_payload(), ensure_ascii=False),
        encoding="utf-8",
    )

    documents = parse_documents(path)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert "JSON 对象正文。" in documents[0].page_content
    assert documents[0].metadata["format"] == "json"
    assert documents[0].metadata["document_index"] == 0
    assert documents[0].metadata["source"] == str(path)


def test_markdown_parser_returns_one_source_document(tmp_path) -> None:
    path = tmp_path / "source.md"
    path.write_text("# 标题\n\nMarkdown 正文。", encoding="utf-8")

    documents = parse_documents(path)

    assert len(documents) == 1
    assert documents[0].page_content.startswith("# 标题")
    assert documents[0].metadata["format"] == "md"
    assert documents[0].metadata["source"] == str(path)


def test_splitter_returns_official_component() -> None:
    splitter = create_splitter(chunk_size=10, chunk_overlap=2)

    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    chunks = splitter.split_documents(
        [Document(page_content="一二三四五六七八九十甲乙")]
    )
    assert chunks
    assert all(len(chunk.page_content) <= 10 for chunk in chunks)


def test_pipeline_only_assembles_components(monkeypatch) -> None:
    embeddings = Mock()
    vector_store = Mock()
    source_document = Document(
        page_content="政策正文",
        metadata={"source": "policies.md", "format": "md"},
    )
    split_document = Document(
        page_content="政策正文",
        metadata={"source": "policies.md", "format": "md"},
    )
    splitter = Mock()
    splitter.split_documents.return_value = [split_document]
    written: list[Document] = []

    monkeypatch.setattr(
        pipeline_module,
        "create_model",
        lambda model_path, device=None, embedding_batch_size=None, query_embedding_batch_size=None: embeddings,
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_vector_store",
        lambda *, embeddings, chroma_path, collection_name=None: vector_store,
    )
    monkeypatch.setattr(
        pipeline_module,
        "parse_documents",
        lambda path: [source_document],
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_splitter",
        lambda *, chunk_size, chunk_overlap: splitter,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_documents",
        lambda vector_store, documents: written.extend(documents),
    )

    pipeline = IngestionPipeline(model_path="model", chroma_path="chroma")
    documents = pipeline.run("policies.json")

    assert documents == written
    assert documents[0].id == "policies-document-0-chunk-0"
    assert documents[0].metadata["chunk_id"] == "policies-document-0-chunk-0"
    splitter.split_documents.assert_called_once_with([source_document])
    pipeline.close()
    embeddings.close.assert_called_once_with()


def test_vectorize_document_uses_stable_uuid_and_object_metadata(monkeypatch) -> None:
    embeddings = Mock()
    vector_store = Mock()
    splitter = Mock()
    splitter.split_documents.return_value = [
        Document(page_content="唯一事实", metadata={})
    ]
    deleted: list[str] = []
    written: list[Document] = []

    monkeypatch.setattr(
        pipeline_module,
        "create_model",
        lambda model_path, device=None, embedding_batch_size=None, query_embedding_batch_size=None: embeddings,
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_vector_store",
        lambda *, embeddings, chroma_path, collection_name=None: vector_store,
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_splitter",
        lambda *, chunk_size, chunk_overlap: splitter,
    )
    monkeypatch.setattr(
        pipeline_module,
        "delete_document_chunks",
        lambda vector_store, document_uuid: deleted.append(document_uuid) or 0,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_documents",
        lambda vector_store, documents: written.extend(documents),
    )

    document_uuid = "550e8400-e29b-41d4-a716-446655440000"
    source = KnowledgeDocument(
        document_uuid=document_uuid,
        content="# 测试知识\n\n唯一事实",
        original_filename="knowledge.txt",
        original_object_key=f"{document_uuid}/original/knowledge.txt",
        converted_object_key=f"{document_uuid}/converted/content.md",
        original_sha256="a" * 64,
        converted_sha256="b" * 64,
    )

    pipeline = IngestionPipeline(model_path="model", chroma_path="chroma")
    chunks = pipeline.vectorize_document(
        source,
        chunk_size=300,
        chunk_overlap=50,
    )

    assert chunks == written
    assert chunks[0].id == f"{document_uuid}:chunk:0"
    assert chunks[0].metadata["document_uuid"] == document_uuid
    assert chunks[0].metadata["original_object_key"].endswith(
        "/original/knowledge.txt"
    )
    assert chunks[0].metadata["converted_object_key"].endswith(
        "/converted/content.md"
    )
    assert deleted == [document_uuid]
    pipeline.close()
