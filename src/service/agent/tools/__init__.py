"""提供服务 Agent 使用的工具包。"""

from langchain_core.tools import tool

from .retrieve import (
    BidCredentialRetrieveInput,
    BidRecordInput,
    CredentialFileInput,
    FileLookupInput,
    KnowledgeFileInput,
    RetrieveInput,
    RetrievalToolContext,
    VectorSearchClient,
    find_ingested_files,
    get_bid_record,
    list_ingested_files,
    list_bid_records,
    read_credential_file,
    read_knowledge_file,
    retrieve,
    retrieve_bid_credentials,
    retrieve_bid_materials,
)


@tool
def exact_paragraph_search(chapter_no: int, paragraph_no: int) -> str:
    """精确定位用户明确指定章节号和段落号的文档内容。"""

    print("模型调用工具：exact_paragraph_search", flush=True)
    return f"占位结果：第{chapter_no}章第{paragraph_no}段"


@tool
def semantic_search(query: str) -> str:
    """根据用户问题检索语义相关内容。"""

    print("模型调用工具：semantic_search", flush=True)
    return f"占位结果：语义检索 {query}"


__all__ = [
    "BidCredentialRetrieveInput",
    "BidRecordInput",
    "CredentialFileInput",
    "FileLookupInput",
    "KnowledgeFileInput",
    "RetrieveInput",
    "RetrievalToolContext",
    "VectorSearchClient",
    "exact_paragraph_search",
    "find_ingested_files",
    "get_bid_record",
    "list_ingested_files",
    "list_bid_records",
    "read_credential_file",
    "read_knowledge_file",
    "retrieve",
    "retrieve_bid_credentials",
    "retrieve_bid_materials",
    "semantic_search",
]
