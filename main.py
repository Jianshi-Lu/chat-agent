"""FastAPI 服务入口和当前的最小演示接口。

Uvicorn 加载 ``main:app`` 后，FastAPI 根据路由表接收 HTTP 请求，调用对应
的异步处理函数，并把函数返回的字典序列化为 JSON 响应。当前入口只用于
验证服务启动和基础路由链路，尚未接入知识文档 Loader、LangChain、MinIO
或 Java 任务接口。
"""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    """返回根路径的服务存活信息。

    输入：无路径参数和请求体。
    内部逻辑：直接构造演示用消息，不访问文件、数据库或外部服务。
    返回：包含 ``message`` 字段的字典，FastAPI 将其序列化为 JSON。
    """

    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    """根据路径参数生成演示用的 JSON 问候消息。

    输入：路径参数 ``name``，由 FastAPI 从 URL 中提取并按 ``str`` 处理。
    内部逻辑：把 ``name`` 插入固定的问候模板，不调用其他模块或外部服务。
    返回：包含 ``message`` 字段的字典，FastAPI 将其序列化为 JSON。
    """

    return {"message": f"Hello {name}"}
