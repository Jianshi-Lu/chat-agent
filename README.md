# chat-agent

招投标问答 Agent。它负责对话模型、LangGraph 编排、MySQL Checkpointer、上下文压缩
和检索工具调用；切分、Embedding、ChromaDB 持久化与实际检索由独立
vector-service 负责。

## 运行

```bash
uv sync
uv run python -m src.api.server
```

默认监听 `127.0.0.1:8082`，并要求 vector-service 已在 `127.0.0.1:8083` 就绪。
常用配置：

- `VECTOR_SERVICE_BASE_URL` / `VECTOR_SERVICE_TIMEOUT`：向量检索服务；
- `CHAT_AGENT_CHECKPOINT_DATABASE_URL`：LangGraph Checkpointer 独立 MySQL；
- `OMLX_API_KEY`：本地 OpenAI-compatible 对话模型认证信息；
- `CHAT_AGENT_LOG_LEVEL`：服务日志级别。

HTTP 服务只公开 `/health` 和 `/chat`。Java 负责平台会话、消息、幂等请求和角色边界，
并把 `conversationId` 作为稳定 `thread_id` 传入；chat-agent 不直接接收文件上传，
也不维护文件生命周期状态。Agent 可以通过 vector-service 的内部工具列出已入库文件、
按原文件名匹配知识文件或资信文件，并按稳定 UUID 读取正文；资信文件读取同时返回
关联投标记录，供模型完成跨文件、跨投标的综合分析。

## 验证

```bash
uv run pytest
```
