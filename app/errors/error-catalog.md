# 应用错误代码目录

本目录与 Java 项目的 `AppErrorCode` 对齐。Python 代码按业务模块复用
`app.errors.exceptions` 中的错误对象；`code` 字段作为跨服务稳定标识，不能
因为实现重构而复用或改变含义。具体失败逻辑通过异常的 `message`、`stage`
和 `details` 描述，不为每个校验分支单独创建错误对象。

## 错误代码与模块映射

| Python 错误对象 | 错误代码 | 出错模块 | 主要阶段 | HTTP 状态 | 是否可重试 |
|---|---|---|---|---:|---|
| `KNOWLEDGE_DOCUMENT_LOAD_FAILED` | `KNOWLEDGE_INGESTION_AGENT.DOCUMENT_LOAD_FAILED` | `app.archive.knowledge.loader` | 由 `stage` 和 `message` 说明 | 500 | 否 |
| `KNOWLEDGE_RETRIEVAL_FAILED` | `KNOWLEDGE_INGESTION_AGENT.RETRIEVAL_FAILED` | `app.archive.retrieval` | 由 `stage` 和 `message` 说明 | 500 | 否 |
| `CHAT_INPUT_INVALID` | `KNOWLEDGE_INGESTION_AGENT.CHAT_INPUT_INVALID` | 后续输入校验模块 | 由 `stage` 和 `message` 说明 | 400 | 否 |
| `CHAT_AGENT_UNAVAILABLE` | `KNOWLEDGE_INGESTION_AGENT.CHAT_AGENT_UNAVAILABLE` | `app.service` | 由 `stage` 和 `message` 说明 | 503 | 是 |
| `CHAT_AGENT_FAILED` | `KNOWLEDGE_INGESTION_AGENT.CHAT_AGENT_FAILED` | `app.service` | 由 `stage` 和 `message` 说明 | 502 | 是 |

## 维护规则

1. 新增业务模块时，优先按模块定义一个稳定错误对象，再补充本表。
2. 具体失败原因写入安全的 `message` 和 `details`；`stage` 用于描述调用链阶段。
3. 错误字符串、默认消息和可重试语义确定后不得随意复用或改变。
4. `module`、`operation` 和 `stage` 描述具体调用位置，不替代错误代码。
5. 原始异常、堆栈、绝对路径和凭据只进入内部日志，不直接返回给 Java 或前端。
6. HTTP 状态码属于接口边界信息；后台任务失败时，应将错误代码和安全消息写入任务状态。
