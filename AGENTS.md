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

截至当前工作区，项目已经完成本地政策文档基础能力和可替换检索器公共契约，当前仍处于检索实现阶段：

```text
knowledge-ingestion-agent/
├── main.py                         # FastAPI 应用入口，目前只有演示路由
├── app/
│   ├── errors/                     # AppErrorCode、AppException 和错误代码
│   ├── knowledge/                  # PolicyDocument、JSON Loader 和知识包边界
│   └── retrieval/                  # Retriever、RetrievedDocument 公共契约
├── data/
│   ├── policies.json               # 当前 97 条本地政策/资讯文档
│   └── interference_ids.txt        # 干扰数据标识，仅作为测试资料
├── tests/
│   ├── __init__.py
│   └── test_loader.py              # 当前 Loader smoke test
├── pyproject.toml                  # 项目元数据和直接依赖
├── uv.lock                         # uv 锁定的依赖解析结果
├── test_main.http                  # 演示路由的 HTTP 请求示例
└── AGENTS.md                       # 本文件
```

当前没有：

- 业务 Service、Repository、Mapper、数据库实体或数据库定义；
- README、应用配置文件、`.env.example`、Dockerfile 或 CI 配置；
- `/policy-ingestion/jobs` 业务路由；
- Keyword、BM25、Vector、Hybrid 具体检索器及 RetrieverRouter；
- LangChain 调用链、MinIO、MarkItDown、Embedding、向量数据库、爬虫、任务队列或持久化任务表实现。

当前 `master` 已包含以下检索基础提交：

- `c8e35a1 feat(retrieval): 建立可替换检索器公共契约`；
- `bacb8a5 refactor(retrieval): 将检索能力提升为独立应用包`。

后续默认沿 `master` 单线开发，通过小粒度提交区分不同检索实现；只有在高风险实验、依赖冲突或多人并行时才临时创建分支。

## 3. 当前技术栈与运行方式

### 3.1 已安装/声明的技术

`pyproject.toml` 当前内容表明：

| 项目 | 当前事实 | 作用 |
|---|---|---|
| Python | `>=3.13` | 项目运行时下限，没有上限约束 |
| 包管理 | uv | 依赖解析、虚拟环境和锁文件管理 |
| Web | FastAPI `>=0.140.7` | HTTP API |
| ASGI 服务 | Uvicorn `>=0.51.0` | 启动 FastAPI 应用 |

`uv.lock` 当前锁定了项目声明的直接依赖及其传递依赖。新增或移除依赖必须通过 `pyproject.toml` 和 `uv.lock` 一起维护，不要只手工修改锁文件。

当前 `pyproject.toml` 已声明 `langchain`、`langchain-openai` 和 `pydantic-settings`，但代码尚未接入 LangChain 调用链、模型服务或配置类。MinIO SDK、MarkItDown、向量数据库客户端、Embedding 模型客户端和爬取工具仍属于计划引入能力，具体包和版本待对应阶段确定。

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

### 3.4 当前知识与检索包边界

`app.knowledge` 和 `app.retrieval` 采用单向依赖：

```text
app.knowledge
  ├── PolicyDocument
  └── load_documents()
          ↑
          │ 具体 Retriever 接收其返回值
app.retrieval
  ├── Retriever
  ├── RetrievedDocument
  └── 后续 Keyword / Vector / Hybrid 实现
```

当前已经实现：

- `app.knowledge.schemas.PolicyDocument`：定义字段契约并执行单条文档校验；
- `app.knowledge.loader.load_documents()`：读取 UTF-8 JSON，校验顶层数组、文档字段和全局 ID 唯一性；
- `app.retrieval.Retriever`：规定 `build(documents)` 和 `retrieve(query, top_k)` 两阶段生命周期；
- `app.retrieval.RetrievedDocument`：统一各检索器返回的文档字段和内部排序分数。

检索公共包只定义调用边界，不读取 `data/policies.json`，不读取或过滤 `data/interference_ids.txt`，不实现具体相关度算法，不执行分块、向量化或大模型调用。`score` 只在同一种检索算法内部用于排序，不可直接跨算法比较。

当前检索实现顺序为：

```text
Keyword → Vector → Hybrid → BM25（可选词法基线）
```

“选择哪种检索算法”和“向模型提供完整正文还是片段”属于两个职责：前者由后续 `RetrieverRouter` 决定，后者由独立 `ContextBuilder` 决定。

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

当前已经引入 pytest，并存在 Loader smoke test；正式 API、检索器和外部依赖测试尚未补齐。新增业务时至少应补充：

- Loader/模型测试：默认数据加载、字段校验、异常和数据边界；
- Retriever 单元测试：build 生命周期、查询、Top-K、稳定排序、空结果和异常；
- API 契约测试：启动任务、查询任务、参数错误、未知 job；
- 单元测试：文档 ID/指纹、去重、版本判断、路径生成、状态收敛；
- 存储适配测试：MinIO 客户端应通过接口 mock 或测试容器隔离；
- 外部来源和模型调用测试：使用 stub，不在测试中依赖真实网站或真实凭据；
- 失败恢复测试：单文档失败、部分失败、进程重启/超时恢复和重复任务。

当前仓库可以执行的基础验证包括：

```text
uv run pytest
.venv/bin/pytest -q             # 当本机 uv 缓存权限导致 uv run 无法执行时的本地替代
GET http://127.0.0.1:8000/
GET http://127.0.0.1:8000/hello/User
```

当前 Loader smoke test 预期能够加载 97 条文档。实现新的 Retriever 时，必须补充对应的功能测试；正式 Benchmark 等至少两种检索方案稳定运行后再建立，不作为第一个检索器的前置条件。

新增依赖和命令优先使用 uv，例如通过 `uv add` 更新 `pyproject.toml` 和 `uv.lock`，通过 `uv run` 执行项目命令。不要提交 `.venv`、`__pycache__`、日志、构建产物或缓存。

## 8. 已实现、未完成和待确认

### 8.1 已实现

- uv 项目元数据：名称 `knowledge-ingestion-agent`、版本 `0.1.0`、Python `>=3.13`；
- FastAPI 应用对象 `main:app`；
- Uvicorn 运行依赖；
- `/` 和 `/hello/{name}` 两个脚手架演示路由；
- `uv.lock` 依赖锁文件；
- `app.errors` 模块级 `AppErrorCode`、`AppException` 和知识文档加载错误代码；
- `app.knowledge.schemas.PolicyDocument` 文档模型；
- `app.knowledge.loader.load_documents()` 本地 JSON Loader；
- `data/policies.json` 当前 97 条政策/资讯文档及保留干扰数据；
- `app.retrieval.Retriever` 公共 Protocol；
- `app.retrieval.RetrievedDocument` 统一检索结果模型；
- Java 侧已存在的启动/查询任务 DTO 契约，可作为后续 Python API 实现依据；
- Loader smoke test，当前能够验证正式数据加载结果。

### 8.2 未完成

- Keyword、Vector、Hybrid 和可选 BM25 Retriever；
- RetrieverRouter 和 Context Builder；
- 正式 Benchmark；
- LangChain 固定问答链、结构化 Agent 和模型调用配置；
- Python 查询 API；
- `/policy-ingestion/jobs` 的 POST/GET API；
- 任务状态存储、后台执行模型、并发限制、重试和恢复；
- 信息源配置和抓取器；
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

当前 MileStone1 Demo 按“先完成可替换检索链路，再接模型和接口”的顺序推进；长期 ingestion 业务仍按 Java 调度、Python 执行的边界实现：

1. 实现 `app.retrieval.keyword.KeywordRetriever`，完成词表、文本格式化、评分、Top-K 和功能测试；
2. 实现 Vector Retriever，先对当前短文档验证整篇 Embedding，长文档再引入分块；
3. 实现 Hybrid Retriever，通过排名融合组合 Keyword 和 Vector；
4. 需要词法对照时再实现 BM25；
5. 增加 RetrieverRouter 和 Context Builder，拆分算法选择与全文/片段组装；
6. 至少两种检索方式稳定后建立正式 Benchmark；
7. 接入 LangChain 固定问答链和结构化 RetrievalPlan，保留无候选直接拒答闸门；
8. 增加 Python 查询 API，再与 Java 查询链路联调；
9. 最后实现 `/policy-ingestion/jobs`、信息源抓取、MinIO、任务状态和失败恢复等长期 ingestion 能力。

每一步都应先写契约/状态/失败边界，再引入对应依赖；不要先搭“自由 Agent”而缺少可恢复的任务状态和持久化事实。

## 10. 提交规范

提交应保持一个提交对应一个可独立理解的逻辑变更。提交标题使用：

```text
<type>(<scope>): <简短中文结论>
```

正文尽可能说明：

- 问题：原有结构或行为造成了什么问题；
- 调整：本次具体修改了哪些文件/边界/调用链；
- 调用边界：输入、输出、上下游关系和未涉及的职责；
- 当前范围：明确哪些规划能力仍未实现；
- 验证：执行过的命令、关键结果和已知限制。

只看提交日志也应能够理解本次变更解决了什么问题。除非用户明确要求，不执行 `git push`；不在提交中写入凭据、个人隐私或真实内部地址。

## 11. Python 注释与文档字符串规范

Python 模块、包和公开类/函数的注释与文档字符串统一使用中文；Python、FastAPI、Pydantic、LangChain、MinIO 等技术名词和 API 名称保留英文，避免为了翻译而失去可检索性。注释服务于理解代码，不重复显而易见的语法。

### 11.1 注释必须说明什么

- 模块级文档字符串：说明模块的职责边界、主要输入输出，以及它在当前调用链中的位置；同时明确尚未接入的关键外部能力，避免把规划误认为现状。
- 包级文档字符串：说明包负责的领域和对外提供的能力，并指出上层服务应如何依赖该包。
- 公开类/函数：说明目的、作用、输入、输出和失败方式。数据模型还要说明字段契约、类型转换和原始数据是否保留。
- 关键内部函数或校验器：说明校验/转换发生的时机、原因、返回值以及是否会改写输入。简单的一行代理函数不强制添加重复注释。
- 中间件或外部依赖：在首次接入的模块级或公开对象文档中说明基本用法、核心原理、输入输出边界和调用/流转链路。例如：FastAPI 接收 HTTP 请求并序列化响应，Pydantic 在模型实例化时校验并转换字段，LangChain 编排步骤，MinIO 通过对象 key 读写原始或派生文件。

### 11.2 篇幅与写法

- 以“目的 → 输入 → 处理/依赖 → 输出 → 失败边界”为最小说明链路，保持简洁；只有跨模块、异步任务或外部依赖才展开中间流转。
- 优先使用中文完整句子；必要的命令、路径、字段名、类型名和状态值使用代码格式标记。
- 注释应描述当前真实行为，不把待实现的 LangChain、MinIO、向量化、调度或持久化写成已实现能力。
- 不在注释、文档字符串或示例中写入密码、API Key、Token、个人隐私、真实内部地址或其他凭据。
- 修改模块职责、接口输入输出、外部依赖或调用链时，同步更新相关中文注释；新增 Python 模块不得保留无说明的英文模板文档字符串。
