"""提供一个在终端中调用本地 OMLX Agent 的最小对话入口。"""

from app.errors import AppException
from app.service.client import ChatClient


def main() -> None:
    """循环读取用户输入，并实时打印 Agent 回答。"""

    try:
        client = ChatClient()
    except AppException as exc:
        print(f"对话客户端初始化失败：{exc}")
        return

    print("已启动本地 OMLX 对话入口。输入 exit、quit 或 退出结束。")

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
