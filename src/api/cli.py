"""本地命令行入口。

本模块直接创建并复用 ``Runtime``，读取终端输入后调用
``Runtime.answer``；它不创建 FastAPI 应用，也不启动 Uvicorn。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from src.errors import AppException
from src.service.runtime import Runtime


_EXIT_COMMANDS = frozenset({"/exit", "/quit", "exit", "quit"})


def _build_parser() -> argparse.ArgumentParser:
    """构造本地 CLI 参数解析器。"""

    parser = argparse.ArgumentParser(
        description="运行本地政策知识问答 CLI。",
    )
    parser.add_argument(
        "-q",
        "--question",
        help="只处理一次问题；未提供时进入交互模式。",
    )
    parser.add_argument(
        "--thread-id",
        required=True,
        help="对话线程标识；必须显式提供，不能使用隐式默认线程。",
    )
    return parser


def _print_error(exc: AppException) -> None:
    """将应用异常转换为适合 CLI 阅读的错误信息。"""

    print(
        f"[{exc.error_code.code}] {exc.public_message}",
        file=sys.stderr,
    )


def _run_interactive(runtime: Runtime, *, thread_id: str) -> None:
    """运行本地交互式问答循环。"""

    print("本地政策问答 CLI 已启动，输入 /exit 或 /quit 退出。")
    while True:
        try:
            question = input("你> ").strip()
        except EOFError:
            print()
            return

        if not question:
            continue
        if question.lower() in _EXIT_COMMANDS:
            return

        try:
            answer = runtime.answer(question, thread_id=thread_id)
        except AppException as exc:
            _print_error(exc)
            continue
        print(f"助手> {answer}")


def main(argv: Sequence[str] | None = None) -> int:
    """启动本地 CLI，并在退出时释放 Runtime 资源。"""

    args = _build_parser().parse_args(argv)
    runtime: Runtime | None = None
    try:
        runtime = Runtime()
        if args.question is not None:
            print(runtime.answer(args.question, thread_id=args.thread_id))
        else:
            _run_interactive(runtime, thread_id=args.thread_id)
        return 0
    except KeyboardInterrupt:
        print()
        return 130
    except AppException as exc:
        _print_error(exc)
        return 1
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
