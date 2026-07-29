"""FastAPI 服务入口和当前的最小演示接口。

本模块由 Uvicorn 作为 ASGI 应用加载：启动命令定位到 ``main:app`` 后，
FastAPI 根据路由装饰器把 HTTP 请求分发给下面的异步函数，并将返回的字典
序列化为 JSON 响应。当前接口只用于确认服务可启动和路由链路可用，尚未接入
知识文档加载、LangChain、MinIO 或 Java 任务接口。
"""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    """返回服务存活信息；输入为空，输出为简单 JSON 消息。"""

    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    """接收路径中的 ``name``，返回包含该名称的 JSON 问候消息。"""

    return {"message": f"Hello {name}"}
