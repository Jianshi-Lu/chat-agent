"""已归档的无状态本地 OMLX Agent 终端对话入口。"""

from archive.client import ChatClient
from src.errors import AppException


def main() -> None:
    """循环读取用户输入，并实时打印归档 Agent 的回答。"""

    try:
        client = ChatClient()
    except AppException as exc:
        print(f"对话客户端初始化失败：{exc}")
        return

    print("已启动归档的本地 OMLX 对话入口。输入 exit、quit 或 退出结束。")

    while True:
        try:
            message = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n对话结束。")
            return

        if message.lower() in {"exit", "quit", "q", "退出"}:
            print("对话结束。")
            return
        if not message:
            continue

        print("助手：", end="", flush=True)
        try:
            for chunk in client.stream_response(message):
                print(chunk, end="", flush=True)
            print()
        except AppException as exc:
            print(f"\n调用失败：{exc}")


if __name__ == "__main__":
    main()
