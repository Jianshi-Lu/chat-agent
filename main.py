"""兼容启动脚本，正式 HTTP 入口位于 ``src.service.__main__``。"""

from src.service.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
