"""知识入库任务入口：读取配置、创建 Pipeline、执行并输出结果。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from configuration import get_ingestion_settings
from src.errors import AppException

from .pipeline import IngestionPipeline


def _build_parser() -> argparse.ArgumentParser:
    """创建入库任务的命令行参数。"""

    parser = argparse.ArgumentParser(
        description="读取 Markdown/JSON 源资料并写入 Chroma collection。"
    )
    parser.add_argument(
        "--source-path",
        "--policies-path",
        dest="source_path",
        type=Path,
        help="源资料路径，支持 .md/.markdown/.json；保留 --policies-path 兼容旧命令。",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--chroma-path", type=Path)
    parser.add_argument("--device")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="可选，覆盖 configuration.config 中的 chunk_size。",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="可选，覆盖 configuration.config 中的 chunk_overlap。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """读取任务参数和默认配置，执行一次知识入库任务。"""

    args = _build_parser().parse_args(argv)
    settings = get_ingestion_settings()
    pipeline: IngestionPipeline | None = None

    try:
        pipeline = IngestionPipeline(
            model_path=args.model_path or settings.embedding_model_path,
            chroma_path=args.chroma_path or settings.chroma_path,
            collection_name=settings.collection_name,
            embedding_batch_size=settings.embedding_batch_size,
            query_embedding_batch_size=settings.query_embedding_batch_size,
            device=args.device,
        )
        documents = pipeline.run(
            args.source_path or settings.source_path,
            chunk_size=(
                settings.chunk_size if args.chunk_size is None else args.chunk_size
            ),
            chunk_overlap=(
                settings.chunk_overlap
                if args.chunk_overlap is None
                else args.chunk_overlap
            ),
        )
        stored_count = len(pipeline.vector_store.get(include=[]).get("ids", []))
        print(
            f"知识入库完成：本次生成 {len(documents)} 个文档块，"
            f"collection 当前共 {stored_count} 个文档块。",
            flush=True,
        )
        return 0
    except AppException as exc:
        print(
            f"[{exc.error_code.code}] {exc.public_message}",
            file=sys.stderr,
        )
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
