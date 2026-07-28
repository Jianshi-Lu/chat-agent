# AGENTS.md

> `knowledge-ingestion-agent` 的 Agent 工作上下文与工程约定。
> 本文件按当前工作区源码和已确定的 Java 对接方向整理。
> “当前已实现”和“目标设计/待确认”必须区分，不要把规划中的 LangChain、MinIO、向量化或任务调度写成现有能力。
> 禁止在代码、文档、日志或提交中写入密码、API Key、Token、个人隐私、真实内部地址或其他凭据。

## 1. 项目定位与边界

这是一个独立的 Python 服务项目，目标是承载综合资讯/政策知识的获取、处理和编排流程，为 Java 招投标平台提供异步任务接口。

已确定的目标职责边界是：

```text
Java 平台
  ├── 保存固定配置（例如信息源）
  ├── 定时或人工触发任务
  └── 调用 Python 接口、查询任务状态

knowledge-ingestion-agent
  ├── 使用 LangChain 编排资讯获取与处理步骤
  ├── 抓取/读取信息源
  ├── 去重和版本判断
  ├── 原始内容写入 MinIO
  ├── 转换为 Markdown 并写入 MinIO converted 路径
  ├── 后续分块、向量化并写入向量库
  └── 返回任务状态和每篇文档结果
```

当前仓库只实现 Python 服务自身；Java 后端、MinIO、文档转换服务、数据库、Embedding 服务和向量数据库都是外部依赖。当前代码尚未实现上述业务闭环。

### 1.1 与现有 Java 平台的边界

Java 项目位于同级目录 `../smart-bid-platform-be`，其中已经实现了 `KnowledgeIngestionAgentClient`。该客户端是本项目当前最重要的外部接口契约来源：

- Java 负责调用 Python，不要求 Python 调用 Java 数据库接口；
- Python 负责资讯采集任务内部的步骤编排和外部存储交互；
- Java 侧当前设计为“启动任务 + 查询任务状态”，而不是同步等待整条采集链路完成；
- 调度器最终放在 Java 平台，Python 暂不承担定时触发职责；
- LangChain 是已确定的编排技术，当前方向不使用 LangGraph。

## 2. 当前仓库事实

截至当前工作区，项目是刚初始化的 FastAPI/uv 脚手架：

```text
knowledge-ingestion-agent/
├── main.py          # 当前唯一 Python 应用入口，只有演示路由
├── pyproject.toml   # 项目元数据和直接依赖
├── uv.lock          # uv 锁定的依赖解析结果
├── test_main.http   # 两个演示路由的 HTTP 请求示例
└── AGENTS.md        # 本文件
```

当前没有：

- 其他 Python package、Service、Repository、Mapper、数据库实体或数据库定义；
- README、应用配置文件、`.env.example`、Dockerfile、CI 配置或正式测试目录；
- `/policy-ingestion/jobs` 业务路由；
- LangChain、MinIO、MarkItDown、Embedding、向量数据库、爬虫、任务队列或持久化任务表实现。

项目 Git 仓库当前尚无提交；工作区中已有的 IDE/脚手架文件属于现有工作区状态，后续 Agent 不应为了整理本文件而修改或清理它们。

## 3. 当前技术栈与运行方式

### 3.1 已安装/声明的技术

`pyproject.toml` 当前内容表明：

| 项目 | 当前事实 | 作用 |
|---|---|---|
| Python | `>=3.13` | 项目运行时下限，没有上限约束 |
| 包管理 | uv | 依赖解析、虚拟环境和锁文件管理 |
| Web | FastAPI `>=0.140.7` | HTTP API |
| ASGI 服务 | Uvicorn `>=0.51.0` | 启动 FastAPI 应用 |

`uv.lock` 当前解析出的直接依赖版本为 FastAPI `0.140.7` 和 Uvicorn `0.51.0`，其余为框架传递依赖。新增依赖必须通过 `pyproject.toml` 和 `uv.lock` 一起维护，不要只手工修改锁文件。

LangChain、MinIO SDK、MarkItDown、向量数据库客户端、Embedding 模型客户端和爬取工具均属于“计划引入、当前未安装”，具体包和版本待实现阶段根据接口和部署方式确定。

### 3.2 当前应用入口

`main.py` 当前创建一个无配置的 `FastAPI()` 实例 `app`，提供：

| Method | Path | 当前行为 |
|---|---|---|
| `GET` | `/` | 返回 `{"message": "Hello World"}` |
| `GET` | `/hello/{name}` | 返回带路径参数的问候消息 |

仓库没有正式启动脚本或 README。依据当前入口，开发时可使用以下命令启动，但该命令目前只是标准运行方式，不代表已存在正式部署约定：

```bash
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

`test_main.http` 使用 `http://127.0.0.1:8000` 验证上述两个演示路由。

### 3.3 配置现状

当前没有 Python 应用配置类或配置文件。后续配置应至少区分：

- 服务监听地址和端口；
- MinIO endpoint、bucket 和非敏感对象路径配置；
- 信息源配置及其启停/抓取策略；
- LangChain/模型服务地址和超时；
- Embedding 和向量数据库连接配置；
- 任务超时、并发、重试和保留策略。

凭据必须通过环境变量或部署平台 Secret 注入，不能写进 `pyproject.toml`、源码、测试请求或 `AGENTS.md`。

## 4. Java 对接契约

Java 当前客户端文件：

```text
../smart-bid-platform-be/src/main/java/io/gitee/lujianshi/bidwinningbackend/knowledgeingestionagent/client/KnowledgeIngestionAgentClient.java
```

它使用 `RestClient`，基础地址来自 Java 配置 `knowledge-ingestion-agent.python-base-url`，当前默认值是 `http://127.0.0.1`。而本 Python 脚手架测试端口是 `8000`，因此本地联调必须显式配置为类似 `http://127.0.0.1:8000`；当前 Java 默认值与脚手架端口并不自动匹配，这是已知集成配置边界。

### 4.1 启动任务

Java 期望调用：

```text
POST /policy-ingestion/jobs
Content-Type: application/json
```

请求体字段来自 `KnowledgeIngestionJobRequest`：

| 字段 | Java 类型 | 语义 |
|---|---|---|
| `jobId` | `String` | Java 侧生成/传入的任务标识，Python 应保持可追踪 |
| `windowStart` | `LocalDateTime` | 本次采集时间窗口起点 |
| `windowEnd` | `LocalDateTime` | 本次采集时间窗口终点 |
| `callbackUrl` | `String` | 可选回调地址；当前 Java 客户端只传递，不处理回调协议 |

当前契约没有携带信息源列表字段。此前约定“信息源属于固定配置，由 Java 平台配置”，但具体是通过 Python 配置文件、任务配置接口还是后续新增字段传递，仍待确认；实现前不要自行改变请求字段含义。

### 4.2 查询任务

Java 期望调用：

```text
GET /policy-ingestion/jobs/{jobId}
```

### 4.3 响应体

Java 期望 `KnowledgeIngestionJobResponse` 结构：

```json
{
  "jobId": "job-xxx",
  "status": "RUNNING",
  "stage": "FETCHING",
  "discoveredCount": 0,
  "indexedCount": 0,
  "skippedCount": 0,
  "failedCount": 0,
  "documents": []
}
```

`status` 和 `stage` 在 Java 中当前是普通字符串，没有共享枚举，因此 Python 首版应先固定并文档化有限状态集合，避免同一语义出现多个拼写。建议的状态方向是 `PENDING/RUNNING/SUCCEEDED/PARTIAL_FAILED/FAILED`，但这些值尚未在当前 Java 契约中正式固化，使用前需要同步确认。

每个 `documents[]` 元素对应 Java 内嵌 `Document`：

| 字段 | 语义 |
|---|---|
| `documentId` | 去重/版本后的文档标识 |
| `title` | 文档标题 |
| `sourceUrl` | 来源地址 |
| `version` | 文档版本；当前版本规则待确认 |
| `originalObjectKey` | MinIO 原始对象 key |
| `convertedObjectKey` | MinIO Markdown 对象 key |
| `chunkCount` | 分块数量；向量化未实现时可为空或按契约约定 |
| `status` | 单文档处理状态 |
| `errorMessage` | 单文档失败原因；不得包含凭据或内部敏感信息 |

Java 客户端只检查 HTTP 调用异常和响应是否为空，当前不会替 Python 校验字段完整性。因此 Python 必须保证响应字段名、JSON 类型和状态语义稳定；接口错误应返回可被 Java 识别的 HTTP 错误，而不是 `200` 搭配不可解析正文。

## 5. 目标业务链路（尚未实现）

### 5.1 任务主链

目标是由 Python 内部完成单个 ingestion job 的闭环，LangChain 负责编排工具/步骤，不能让模型自由绕过必要的持久化和状态更新：

```text
Java 定时任务或人工触发
  ↓ POST /policy-ingestion/jobs
Python 创建/登记 job
  ↓
读取已配置的信息源和时间窗口
  ↓
发现资讯文档
  ↓
规范化元数据、去重、判断版本
  ↓
原始内容写入 MinIO original
  ↓
转换为 Markdown，写入同一文档的 converted
  ↓
切分 Markdown、生成 Embedding、写入向量数据库
  ↓
更新每篇 Document 结果和整个 job 状态
  ↓ GET /policy-ingestion/jobs/{jobId}
Java 获取任务进度/结果
```

其中“信息源发现、抓取方式、去重规则、文档转换、分块和向量化”是 Python 业务职责；Java 不应把这些步骤重新实现一份。Java 负责调度、外部配置边界和对前端暴露平台级结果。

### 5.2 LangChain 的职责边界

已确定使用 LangChain，而不是 LangGraph。当前尚未写任何 LangChain 代码，下面只是实现边界，不是现有 API：

- 将抓取、解析、去重、落盘、转换、分块和向量化封装为可测试的普通 Python 能力；
- 使用 LangChain 组织需要模型/工具参与的处理步骤；
- 对“必须执行”的存储、状态、错误归档步骤使用确定性代码，不把关键数据一致性完全交给模型；
- 每个步骤应能返回结构化结果，便于更新 job 和 document 状态；
- 第一期优先实现固定流程式 Agent；不要因为使用 LangChain 就引入 LangGraph 或完全自主的无限循环 Agent。

### 5.3 存储与知识索引边界

目标存储层建议保持对象路径可追踪且与文档 ID 稳定绑定，例如：

```text
policy/{source}/{documentId}/original/{version-or-filename}
policy/{source}/{documentId}/converted/{version}.md
```

上述路径只是当前方向，具体 bucket、prefix、版本命名和覆盖/追加策略待确认，不能在没有契约的情况下固化为业务事实。原始文档应保留用于回溯；Markdown 是派生物；向量索引应保存 `documentId`、来源、版本、发布时间和 object key 等 metadata，以便回答时返回引用来源。

当前项目没有数据库或向量数据库实现，不能假定 `indexedCount` 已有持久化来源。需要后续明确：任务状态/文档元数据放在哪里，向量库选择什么，重跑时如何幂等，MinIO 和向量索引如何保持可追踪关系。

## 6. 失败、重试和幂等约束（实现前必须保留）

后续实现不能只返回一条“任务失败”字符串，至少要区分：

- job 级失败：请求非法、配置缺失、任务无法创建；
- document 级失败：单个来源抓取、解析、存储、转换或索引失败；
- 外部依赖失败：来源站点、MinIO、模型/Embedding、向量库不可用；
- 部分成功：其他文档已完成，失败文档带 `errorMessage` 并能重试。

重试设计必须遵守：

- 使用稳定 `jobId`/`documentId` 或内容指纹识别重复任务；
- 原始对象和 converted 对象的覆盖/版本策略必须明确；
- 同一文档重复索引不能无限产生重复向量；
- job 状态更新应与文档结果更新保持可解释关系；
- 任务进程重启后不能让任务永久停留在运行态，需有超时、恢复或明确失败策略；
- 对来源站点设置超时、限速和失败隔离，不能让一个站点阻塞整个批次；
- 不记录页面中的敏感信息、请求头凭据或外部服务密钥。

以上机制当前均未实现，属于后续业务设计约束。

## 7. 测试与开发约定

当前只有 `test_main.http`，没有 pytest、测试夹具、集成测试或 CI 配置。新增业务时至少应补充：

- API 契约测试：启动任务、查询任务、参数错误、未知 job；
- 单元测试：文档 ID/指纹、去重、版本判断、路径生成、状态收敛；
- 存储适配测试：MinIO 客户端应通过接口 mock 或测试容器隔离；
- 外部来源和模型调用测试：使用 stub，不在测试中依赖真实网站或真实凭据；
- 失败恢复测试：单文档失败、部分失败、进程重启/超时恢复和重复任务。

当前开发验证只能覆盖：

```text
GET http://127.0.0.1:8000/
GET http://127.0.0.1:8000/hello/User
```

新增依赖和命令优先使用 uv，例如通过 `uv add` 更新 `pyproject.toml` 和 `uv.lock`，通过 `uv run` 执行项目命令。不要提交 `.venv`、`__pycache__`、日志、构建产物或缓存。

## 8. 已实现、未完成和待确认

### 8.1 已实现

- uv 项目元数据：名称 `knowledge-ingestion-agent`、版本 `0.1.0`、Python `>=3.13`；
- FastAPI 应用对象 `main:app`；
- Uvicorn 运行依赖；
- `/` 和 `/hello/{name}` 两个脚手架演示路由；
- `uv.lock` 依赖锁文件；
- Java 侧已存在的启动/查询任务 DTO 契约可作为 Python API 实现依据。

### 8.2 未完成

- `/policy-ingestion/jobs` 的 POST/GET API；
- 任务状态存储、后台执行模型、并发限制、重试和恢复；
- 信息源配置和抓取器；
- LangChain 编排；
- 去重、版本管理和内容规范化；
- MinIO original/converted 持久化；
- Markdown 转换；
- 文档分块、Embedding 和向量数据库写入；
- 与 Java 的完整联调、正式测试、日志/监控和部署脚本；
- 前端政策资讯展示和“已纳入知识库”状态展示。

### 8.3 待确认

- Python job 是内存任务、数据库任务还是外部任务队列；
- job 状态和 document 元数据的持久化位置；
- 信息源列表由 Java 传递还是 Python 配置管理；当前 Java request 没有 source 字段；
- MinIO bucket、prefix、对象 key 版本策略，以及 original/converted 是否覆盖；
- 文档唯一 ID、内容 hash、来源版本和发布时间规则；
- MarkItDown 或其他转换器的选型和 OCR 能力；
- Embedding 模型、向量数据库及 metadata schema；
- callbackUrl 是否真正启用，还是只使用 Java 轮询；
- Python API 端口和 Java `python-base-url` 的最终部署配置；
- 任务超时、并发、限速和失败重试策略；
- 资讯来源的授权、robots、访问频率和内容合规边界。

## 9. 后续 Agent 的实现顺序

在不推翻 FastAPI + uv + 独立 Python 服务选型的前提下，建议按以下里程碑推进：

1. 先把 `/policy-ingestion/jobs` 启动/查询接口和 Java DTO 契约固定，并补充 Pydantic request/response model；
2. 设计 job/document 状态模型和最小持久化方案，先实现可恢复的后台任务骨架；
3. 接入信息源配置与抓取器，完成去重、原始内容落盘 MinIO；
4. 接入 Markdown 转换，保证 original/converted object key 和文档结果可追踪；
5. 使用 LangChain 编排确定性处理链，补齐单文档和批次失败收敛；
6. 接入分块、Embedding 和向量数据库，返回 `indexedCount`/`chunkCount` 的真实值；
7. 与 Java 定时触发、状态查询及错误码联调，再扩展前端展示。

每一步都应先写契约/状态/失败边界，再引入对应依赖；不要先搭“自由 Agent”而缺少可恢复的任务状态和持久化事实。
