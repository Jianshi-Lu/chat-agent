"""HTTP Server 启动入口。

本模块只负责读取 HTTP 配置并启动 Uvicorn。FastAPI 应用和生命周期位于
``src.api.app``，模型、持久化 Chroma 和 LangChain Agent 由 Runtime 管理。
"""

import uvicorn

from configuration import get_http_server_settings
from src.api.app import app


def main() -> int:
    """读取配置并启动单 worker Uvicorn Server。"""

    settings = get_http_server_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
