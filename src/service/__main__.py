"""启动本地 HTTP Server。

本模块只负责读取服务配置并启动 Uvicorn；HTTP 路由和 FastAPI 生命周期位于
``src.api.app``，模型、内存 Chroma 和 RAG Chain 由 Runtime 管理。
"""

import uvicorn

from configuration import get_server_settings
from src.api.app import app


def main() -> int:
    """读取配置并启动单 worker Uvicorn Server。"""

    settings = get_server_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
