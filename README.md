# ScriptAgent

面向电商直播/短视频场景的多 Agent 话术生成系统。项目已经具备完整的前后端链路、鉴权、会话管理、流式生成、质量校验、长期记忆召回与多模型容灾能力，可直接本地运行，也可容器化部署到云服务器。

- 版本演进：`CHANGELOG.md`
- 阿里云部署手册：`DEPLOY_ALIYUN_ECS.md`
- 监控面板手册：`docs/MONITORING.md`
- 架构图与高保真生成提示词：`docs/architecture.mmd`、`docs/architecture_prompt_for_gemini.md`

---

## 一、项目定位

ScriptAgent 解决的是“可持续、多轮可控、可上线运营”的话术生成，不是单次问答 Demo。核心目标：

- 多轮对话中保持意图连续，支持换品/换风格/续写。
- 输出可直接口播，避免提示词回显、半句截断、空结果。
- 线上具备基本工程能力：鉴权、限流、并发锁、checkpoint、降级容灾。

---

## 二、系统架构
详情可看/docs/architecture.mmd
```text
Web UI / API Client
        |
FastAPI API Layer (同步API / 异步任务队列)
(app/auth/sessions/generate/stream/health/metrics/async-task)
        |
    ┌───┴────────────────────────────────┐
    │                                    │
  Kafka Task Queue              Orchestrator (LangGraph / fallback sequential)
  (异步任务处理)                        |
    │                          +-- Intent Agent      (意图识别 + 槽位抽取 + 换品纠偏)
    └──────> Worker            +-- Profile Agent     (缓存 + 数据库 + 通用画像回退)
                               +-- Product Agent     (商品画像 + 长期记忆召回)
                               +-- Script Agent      (Prompt 组装 + 重试 + 兜底 + 清洗)
                               +-- Quality Agent     (敏感词/合规/风格/结构校验)
        |
Infra Services
+-- LLM Client (zhipu/vllm/ollama, retry, circuit-breaker, fallback)
+-- Session Store (memory/redis/sqlite)
+-- Checkpoint Store (memory/redis/sqlite)
+-- Session Lock (local/redis)
+-- Core Rate Limiter (qps + token)
+-- Domain Data Repository (influencer/product sqlite)
+-- Long-Term Memory (memory/elasticsearch, dense+sparse recall)
        |
Protocol Layer (标准化对外协议)
+-- MCP Server (Model Context Protocol): 暴露tools/resources/prompts
+-- A2A Protocol (Agent-to-Agent): 跨框架Agent互操作
```

---

## 三、核心能力设计（平台能力）

### 1) LangGraph 编排与回退执行
- 编排器以 LangGraph 作为主执行引擎，状态由 `WorkflowState` 统一定义。
- 若环境未安装 LangGraph，自动降级到顺序执行路径，保持接口兼容。
- 对外接口保持不变：`handle_request`、`handle_stream`。

### 2) 对话快速恢复机制（Checkpoint 独立存储）
- Checkpoint 从 `SessionContext.workflow_snapshot` 拆分为独立存储层，支持 `Memory/Redis/SQLite`。
- 每次写入带版本号（`version` 自增）和审计字段：
  - `trace_id`
  - `status`
  - `checksum`
  - `created_at`
- 支持：
  - 最新快照读取（快速恢复）
  - 历史版本回放（故障排查）
  - 历史列表查询（审计）
- 同请求去重缓存（`REQUEST_DEDUP_ENABLED`）可在完成态快速返回，避免重复调用模型。

### 3) 并发控制（多实例一致）
- 会话锁支持双后端：
  - `SessionLockManager`（进程内）
  - `RedisSessionLockManager`（分布式锁，`SET NX PX + Lua` 原子释放）
- 分布式锁异常时自动降级到本地锁，保证服务可用性。
- `/generate` 和 `/generate/stream` 均纳入会话锁，避免同会话并发写冲突。

### 4) 核心接口限流与保护
- `CoreRateLimiter` 支持 `Local/Redis` 两种后端。
- 对租户维度同时实施：
  - `QPS` 限流
  - `Token/min` 限流
- 超限返回 `429`，并附带 `reason` 与 `retry_after_seconds`。

### 5) 生成失败降级（LLM Fallback）
- 主模型调用失败时，`LLMServiceClient` 可自动切换到备用模型。
- 支持同步与流式场景（流式在首段失败时切换）。
- 可配置备用后端类型（`vLLM/Ollama/zhipu`）和备用模型。

### 6) 商品理解与长期记忆召回
- `ProductAgent` 从商品名/卖点/特征构建商品画像，补齐“达人风格 + 商品卖点”联合生成场景。
- 长期记忆检索服务支持 `embedding` 向量化 + 向量检索（`memory/elasticsearch` 后端）。
- 生成成功后自动写回长期记忆，后续请求可按租户、达人、品类、商品名做相似召回。

### 7) 会话记忆规则裁剪与压缩落地
- `SessionManager.save()` 增加 retention policy：
  - 旧轮次规则压缩（保留意图槽位摘要与核心语义）
  - 最大轮次裁剪（防止会话无限膨胀）
  - 生成脚本列表裁剪
- `ScriptGenerationAgent` 已接入 `SessionContextCompressor`，多轮场景会优先使用压缩会话记忆构建 Prompt。

### 8) Kafka异步任务队列
- **问题**：API同步处理导致高峰期P99延迟达8.2s，用户体验差
- **方案**：引入Kafka消息队列实现异步任务处理
  - API立即返回task_id（P99 < 200ms）
  - 后台Worker从Kafka消费任务并异步生成
  - 前端通过轮询或WebSocket获取结果
- **特性**：
  - 支持优先级队列（high/normal/low）
  - 任务状态追踪（pending/processing/completed/failed）
  - 多Worker并发处理（水平扩展）
  - 任务结果持久化到Redis（24小时）
- **使用场景**：
  - 高并发场景下削峰填谷
  - 批量生成任务
  - 长耗时操作（如视频话术生成）
- **示例**：`examples/kafka_async_example.py`

### 9) MCP协议支持
- **背景**：Model Context Protocol是Claude推出的标准化上下文协议
- **实现**：完整的MCP Server，暴露ScriptAgent能力
  - **Tools**：生成话术、质量评估等可调用工具
  - **Resources**：达人画像、商品信息等数据资源
  - **Prompts**：开场话术、结尾话术等模板
- **价值**：
  - 跨语言Agent集成（Node.js前端 → Python后端）
  - 符合Claude/OpenAI生态标准
  - 自动发现能力（类似OpenAPI）
  - 统一错误处理和类型校验
- **对比传统API**：MCP之于AI Agent ≈ OpenAPI之于REST API
- **示例**：`examples/mcp_server_example.py`

### 10) A2A协议支持
- **背景**：Agent-to-Agent标准化通信协议
- **实现**：完整的A2A路由器和消息协议
  - 标准化消息格式（兼容OpenAI/AutoGen/LangChain）
  - Agent注册和自动发现
  - 对话上下文管理
  - 协议互转（A2A ↔ OpenAI ↔ AutoGen）
- **已集成Agent**：
  - IntentAgent、ProfileAgent、ProductAgent
  - ScriptAgent、QualityAgent、Orchestrator
- **价值**：
  - 与其他Agent生态互通（OpenAI Swarm、AutoGen）
  - 跨框架Agent协作
  - 统一错误处理和超时控制
  - 支持工具调用（tool_calls）
- **对比自定义协议**：A2A之于Agent ≈ HTTP之于Web服务
- **示例**：`examples/a2a_protocol_example.py`

## 四、核心模块与功能映射

| 模块 | 文件 | 功能点 |
| --- | --- | --- |
| API 网关层 | `script_agent/api/app.py` | 会话创建、同步/流式生成、checkpoint 查询、健康检查、指标出口 |
| 鉴权层 | `script_agent/api/auth.py` | API Key/JWT 鉴权、租户隔离、基础限流 |
| 编排层 | `script_agent/agents/orchestrator.py` | LangGraph 图编排、状态推进、恢复续跑、去重缓存、checkpoint 写入 |
| 意图识别 | `script_agent/agents/intent_agent.py` | 意图分类、槽位提取、指代/续写消解、澄清判断 |
| 画像层 | `script_agent/agents/profile_agent.py` | 达人画像聚合、缓存利用、风格上下文补全 |
| 商品层 | `script_agent/agents/product_agent.py` | 商品画像构建、卖点补全、长期记忆召回 |
| 生成层 | `script_agent/agents/script_agent.py` | Prompt 构建、垂类话术生成、参数控制 |
| 质检层 | `script_agent/agents/quality_agent.py` | 敏感词、合规、风格一致性校验与重试建议 |
| Skill 扩展层 | `script_agent/skills/` | 意图到技能路由，支持生成/修改/批量等可插拔能力 |
| LLM 服务层 | `script_agent/services/llm_client.py` | 统一 vLLM/Ollama 接口、连接池复用、主备降级 |
| 会话管理层 | `script_agent/services/session_manager.py` | 会话增删改查、序列化、Memory/Redis/SQLite 多后端 |
| 并发控制层 | `script_agent/services/concurrency.py` | 本地锁/Redis 分布式会话锁、超时控制、统计 |
| Checkpoint 层 | `script_agent/services/checkpoint_store.py` | 版本化 checkpoint、回放、审计、存储后端抽象 |
| 限流层 | `script_agent/services/core_rate_limiter.py` | 租户维度 QPS + Token 双限流，支持 Redis 共享配额 |
| 长期记忆层 | `script_agent/services/long_term_memory.py` | 向量化写回与检索召回（Memory/Elasticsearch） |
| 异步任务队列层 | `script_agent/services/kafka_queue.py` | Kafka生产者/消费者、任务状态管理、优先级队列 |
| MCP协议层 | `script_agent/protocols/mcp_server.py` | MCP Server实现、工具/资源/提示词注册 |
| A2A协议层 | `script_agent/protocols/a2a_protocol.py` | A2A路由器、消息协议、跨框架适配 |
| 可观测层 | `script_agent/observability/metrics.py` | 请求、延迟、锁超时、checkpoint 写入、缓存命中等指标 |
| 配置层 | `script_agent/config/settings.py` | 环境变量集中配置，覆盖编排/锁/限流/checkpoint/LLM 主备 |

## 五、核心能力设计（按模块）

### 1) API 层（`script_agent/api/app.py`）
- 同时支持同步生成与 SSE 流式生成，便于前端实时渲染。
- 全链路注入 `trace_id`、状态与时延，便于排障。
- 会话级并发锁保护，避免同会话并发生成造成数据污染。
- 健康接口返回编排、锁、限流、checkpoint、长期记忆等运行状态。

### 2) 鉴权与租户隔离（`script_agent/api/auth.py`）
- 支持 `X-API-Key` 与 `Bearer JWT` 双认证模式。
- 所有核心会话接口按 `tenant_id` 做强隔离。
- 内置登录注册能力（SQLite 用户库），可直接用于内网运营后台。
- 开发环境支持 bypass，降低本地联调门槛。

### 3) 编排中枢（`script_agent/agents/orchestrator.py`）
- LangGraph 声明式状态图 + 顺序回退执行双通道，鲁棒性更好。
- 支持请求去重缓存（完成态快速返回），降低重复请求成本。
- 流式与同步链路统一状态推进、checkpoint 写入与错误语义。
- Skill 路由与 Agent 链路共存，可逐步扩展工具化能力。

### 4) 意图识别 Agent（`script_agent/agents/intent_agent.py`）
- 分级意图识别：快速分类优先，低置信再走 LLM 兜底。
- 多来源槽位合并：规则抽取 + 指代消解 + 上下文继承。
- 自动写入 `_raw_query/requirements`，让生成器拿到原始用户诉求。
- 换品纠偏逻辑完善：自动识别“从旧商品切到新商品”，并将意图纠正到生成链路。

### 5) 达人画像 Agent（`script_agent/agents/profile_agent.py`）
- L1 本地缓存 + L2 缓存 + DB + 通用画像回退，多级容错。
- 支持按达人/品类动态构建画像，不依赖单一数据源。
- 在数据缺失时仍能给出可用风格配置，避免链路中断。

### 6) 商品 Agent（`script_agent/agents/product_agent.py`）
- DB 命中优先，规则知识库兜底，确保商品画像始终可构建。
- 自动融合：商品特征、卖点、合规提醒、目标人群。
- 内置长期记忆召回查询拼装，提升多轮场景下的话术一致性与信息密度。

### 7) 话术生成 Agent（`script_agent/agents/script_agent.py`）
- PromptBuilder 按“角色/风格/场景/商品/历史/记忆/要求”结构化组装。
- 强制最小有效长度、句尾完整性、重复抑制、换品约束校验。
- 主模型多次重试 + 失败后自动走 fallback 模型。
- 输出后置清洗增强：字段回显、提示词回显、占位符、重复句等自动剥离。

### 8) 质量校验 Agent（`script_agent/agents/quality_agent.py`）
- 并行检测敏感词、合规、风格一致性、结构完整性。
- 内置 AC 自动机敏感词扫描，效率和可解释性兼顾。
- 对 prompt-echo、内容过简、句尾不完整、换品违规进行结构化反馈。
- 反馈可回注到重试要求，形成“生成-质检-再生成”闭环。

### 9) LLM 客户端（`script_agent/services/llm_client.py`）
- 统一封装 `zhipu/vllm/ollama`，减少上层分支复杂度。
- 内置重试、断路器、超时分层、fallback 计划。
- 支持流式/同步一致清洗逻辑，显著降低脏输出。
- 连接池复用，减少高并发场景连接开销。

### 10) 会话与状态基础设施

#### 会话存储（`script_agent/services/session_manager.py`）
- `memory/redis/sqlite` 三后端统一抽象。
- 会话裁剪与压缩策略，控制多轮对话体积。

#### Checkpoint（`script_agent/services/checkpoint_store.py`）
- 版本化保存，支持 latest / history / replay。
- 每条记录带 checksum 与状态，便于审计与恢复。

#### 并发锁（`script_agent/services/concurrency.py`）
- 支持 local/redis 分布式会话锁。
- 锁服务异常可降级本地锁，服务不中断。

#### 核心限流（`script_agent/services/core_rate_limiter.py`）
- 租户维度双限流：QPS + token/min。
- local 与 redis 两种配额后端，适配单机与多实例。

### 11) Skill 治理与安全（`script_agent/skills/`）
- SkillRegistry 按意图评分路由，可插拔扩展生成/修改/批量技能。
- 统一 preflight 安全链路：required slots + strict schema + role/tenant allowlist + prompt injection tripwire。
- 工具调用治理能力独立于业务逻辑，适合持续扩展。

### 12) 领域数据仓储（`script_agent/services/domain_data_repository.py`）
- 达人画像与商品画像同库管理，默认 SQLite，部署简单。
- 自动建表与 upsert，便于从 mock 平滑迁移到真实数据。
- Profile/Product Agent 无缝集成，命中后可直接提升生成质量。

### 13) 前端交互（`script_agent/web/`）
- 双栏对话式 UI：会话历史 + 主对话区。
- 支持登录/注册页面与主页面联动。
- 生成链路含空流保护、错误提示、重生成、导出、复制。
- 线上可关闭“连接配置面板”，避免直接暴露调试入口。

---

## 六、功能清单

### 核心功能
- 话术生成：开场话术、卖点介绍、促销话术、种草文案。
- 多轮对话：续写、改风格、换商品、重生成。
- 历史会话：创建、查询、删除。
- 认证能力：注册、登录、API Key/JWT。
- 质量治理：敏感词、合规、结构完整性、换品约束。
- 模型治理：主模型重试、兜底模型切换、回显清洗。

### 高级特性 🚀
- **异步任务队列**：基于Kafka的异步生成，P99延迟<200ms
- **MCP协议**：标准化工具/资源/提示词暴露，跨语言集成
- **A2A协议**：Agent间标准化通信，兼容OpenAI/AutoGen/LangChain
- **跨框架互操作**：与其他Agent生态无缝集成

---

## 七、快速开始（本地）

### 1) 环境要求

- Python `>=3.10`
- 推荐：`pip`、`virtualenv`
- 可选（高级特性）：
  - Kafka（异步任务队列）
  - Redis（任务状态存储）

### 2) 安装与配置

```bash
git clone <your-repo-url>
cd scriptAgent

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 基础依赖
pip install -r requirements.txt

# 可选：安装Kafka支持
pip install aiokafka

cp .env.example .env
```

### 3) 启动服务

#### 基础模式（不含异步队列）
```bash
python -m script_agent.main
# 或
uvicorn script_agent.api.app:app --host 127.0.0.1 --port 8080
```

#### 完整模式（含Kafka异步队列）
```bash
# 1. 启动Kafka（Docker）
docker run -d --name kafka -p 9092:9092 apache/kafka:latest

# 2. 启动API服务
python -m script_agent.main

# 3. 启动Worker（另开终端）
python examples/kafka_async_example.py consumer
```

访问：
- 主页面：`http://127.0.0.1:8080/`
- 登录页：`http://127.0.0.1:8080/auth`
- 健康检查：`http://127.0.0.1:8080/api/v1/health`

### 4) 新特性示例 🚀

```bash
# Kafka异步任务队列示例
python examples/kafka_async_example.py full

# MCP协议服务器示例
python examples/mcp_server_example.py

# A2A协议通信示例
python examples/a2a_protocol_example.py
```

---

## 八、关键环境变量

以 `.env.example` 为准，下面是最常用项。

| 变量 | 说明 | 推荐值 |
| --- | --- | --- |
| `APP_ENV` | 运行环境 | `development` / `production` |
| `LLM_PRIMARY_BACKEND` | 主模型后端 | 开发建议 `zhipu`，生产建议 `vllm` |
| `VLLM_BASE_URL` | vLLM 地址 | 生产必填 |
| `ZHIPU_API_KEY` | 智谱 API Key | 使用 zhipu 或 zhipu fallback 时必填 |
| `SCRIPT_MIN_CHARS` | 最小输出字符数 | `40` |
| `SCRIPT_PRIMARY_ATTEMPTS` | 主模型重试次数 | `2` |
| `LLM_FALLBACK_ENABLED` | 是否启用兜底 | `true` |
| `LLM_FALLBACK_BACKEND` | 兜底后端 | `zhipu` / `ollama` / `vllm` |
| `DOMAIN_DATA_ENABLED` | 是否启用领域数据仓库 | `true` |
| `DOMAIN_DATA_DB_PATH` | 领域数据 SQLite 路径 | `domain_data.db` |
| `AUTH_DB_PATH` | 用户库 SQLite 路径 | `auth_users.db` |
| `FRONTEND_EXPOSE_CONNECTION_PANEL` | 是否暴露前端连接面板 | 生产建议 `false` |

模型默认策略：
- 本地/测试可用小参数模型（`QWEN_MODEL_LOCAL=qwen2.5:0.5b`）
- 生产建议 7B（`QWEN_MODEL_PRODUCTION=qwen2.5:7b` 或 `VLLM_MODEL`）

---

## 九、API 概览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 前端主页面 |
| `GET` | `/auth` | 登录注册页面 |
| `GET` | `/api/v1/frontend-config` | 前端运行时配置 |
| `POST` | `/api/v1/auth/register` | 注册 |
| `POST` | `/api/v1/auth/login` | 登录并获取 token |
| `POST` | `/api/v1/sessions` | 创建会话 |
| `GET` | `/api/v1/sessions` | 列出会话 |
| `GET` | `/api/v1/sessions/{session_id}` | 会话详情 |
| `DELETE` | `/api/v1/sessions/{session_id}` | 删除会话 |
| `POST` | `/api/v1/generate` | 同步生成 |
| `POST` | `/api/v1/generate/stream` | 流式生成（SSE） |
| `GET` | `/api/v1/sessions/{session_id}/checkpoints` | checkpoint 历史 |
| `GET` | `/api/v1/sessions/{session_id}/checkpoints/latest` | 最新 checkpoint |
| `GET` | `/api/v1/sessions/{session_id}/checkpoints/{version}` | 按版本回放 |
| `GET` | `/api/v1/skills` | 技能列表 |
| `GET` | `/api/v1/health` | 健康检查 |
| `GET` | `/metrics` | Prometheus 指标 |

### 快速调用示例

```bash
# 1) 创建会话
curl -sS -X POST http://127.0.0.1:8080/api/v1/sessions \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: tenant_dev' \
  -H 'X-Role: admin' \
  -d '{"influencer_name":"小雅","category":"美妆"}'

# 2) 同步生成
curl -sS -X POST http://127.0.0.1:8080/api/v1/generate \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: tenant_dev' \
  -H 'X-Role: admin' \
  -d '{"session_id":"<SESSION_ID>","query":"请生成一段直播带货开场话术，风格活泼"}'

# 3) 流式生成
curl -N -X POST http://127.0.0.1:8080/api/v1/generate/stream \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: tenant_dev' \
  -H 'X-Role: admin' \
  -d '{"session_id":"<SESSION_ID>","query":"换成卫龙辣条，继续卖点介绍"}'
```

---

## 十、部署

### 1) Docker Compose（推荐）

```bash
cp .env.production.example .env
# 编辑 .env，至少填写 ZHIPU_API_KEY / VLLM_BASE_URL 等

docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml ps
curl -sS http://127.0.0.1:8080/api/v1/health
```

### 2) 阿里云 ECS

详见：`DEPLOY_ALIYUN_ECS.md`

### 3) 监控面板（Prometheus + Grafana）

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml up -d --build
```

访问：
- Prometheus：`http://<server-ip>:9090`
- Grafana：`http://<server-ip>:3000`
- 若公网无法访问，请在云安全组放行 TCP `3000`、`9090`（生产建议仅对内网开放 `9090`）

默认账号：
- 用户名：`admin`
- 密码：`admin123`

更多细节（指标说明、线上安全建议）见：`docs/MONITORING.md`

---

## 十一、测试

```bash
# 全量
pytest -q

# 重点模块
pytest -q tests/test_agents.py
pytest -q tests/test_frontend_e2e.py
pytest -q tests/test_auth_and_history_features.py
pytest -q tests/test_api_real_llm_e2e.py
pytest -q tests/test_llm_client_reliability.py
```

说明：`tests/test_api_real_llm_e2e.py` 为真实模型链路验证，需按测试说明配置环境变量与可用模型服务。

---

## 十二、目录结构（核心）

```text
script_agent/
  api/            # FastAPI 接口、鉴权
  agents/         # Intent/Profile/Product/Script/Quality/Orchestrator
  skills/         # Skill 框架、内置技能、安全治理
  services/       # LLM 客户端、会话/锁/限流/checkpoint/长期记忆/领域数据
                  # + kafka_queue.py (异步任务队列) 🆕
  protocols/      # MCP Server、A2A Protocol 🆕
  models/         # 数据结构与状态定义
  web/            # 前端页面（主页面 + 登录页）
  config/         # 配置中心

examples/         # 使用示例 🆕
  kafka_async_example.py    # Kafka异步队列示例
  mcp_server_example.py     # MCP协议服务器示例
  a2a_protocol_example.py   # A2A协议通信示例

tests/            # 单元/集成/E2E 测试
```

---

## 十三、补充说明

- 项目默认可从 mock/规则能力启动，随后平滑接入真实达人与商品数据。
- 如果你要做线上版本，建议优先完成三件事：
  - 关闭前端连接配置面板（`FRONTEND_EXPOSE_CONNECTION_PANEL=false`）
  - 启用真实鉴权策略（API Key/JWT）
  - 使用 Redis 作为锁与限流后端，避免多实例配额不一致

## 十四、高级特性说明 

### Kafka异步任务队列
**适用场景**：
- 高并发场景（QPS > 50）
- 需要快速响应的API（P99 < 200ms）
- 批量生成任务
- 长耗时操作

**架构优势**：
- API响应延迟降低97%（8.2s → 0.2s）
- 支持水平扩展（增加Worker数量）
- 任务优先级控制
- 故障隔离（Worker失败不影响API）

**性能对比**：
| 模式 | P50延迟 | P99延迟 | 吞吐量 |
|-----|---------|---------|--------|
| 同步模式 | 2.1s | 8.2s | 50 QPS |
| 异步模式 | 150ms | 200ms | 200+ QPS |

**使用方法**：
```python
from script_agent.services.kafka_queue import init_kafka_queue

# 初始化队列
queue = init_kafka_queue(
    bootstrap_servers="localhost:9092",
    result_store=redis_client,
)

# 提交任务
task_id = await queue.submit_task(
    task_id="task_001",
    session_id="sess_001",
    user_id="user_001",
    task_type="generate_script",
    payload={"talent_name": "李佳琦", "product_name": "口红"},
    priority=TaskPriority.HIGH,
)

# 查询状态
status = await queue.get_task_status(task_id)
```
---

### MCP协议支持
**适用场景**：
- 跨语言Agent集成（Node.js、Python、Go）
- 与Claude Desktop集成
- 对外暴露AI能力
- 标准化工具生态

**核心概念**：
- **Tools**：可调用的函数（如`generate_script`、`evaluate_quality`）
- **Resources**：可读取的数据（如达人画像、商品信息）
- **Prompts**：可渲染的模板（如开场话术模板）

**对比传统API**：
| 维度 | 传统REST API | MCP协议 |
|-----|-------------|---------|
| 发现能力 | 需要文档 | 自动发现（类似OpenAPI） |
| 类型校验 | 手动实现 | JSON Schema自动校验 |
| 跨语言 | 需要各语言SDK | 标准协议，任何语言可调用 |
| 生态兼容 | 自定义格式 | Claude/OpenAI生态标准 |

**使用方法**：
```python
from script_agent.protocols import create_scriptagent_mcp_server

# 创建MCP Server
server = create_scriptagent_mcp_server(
    orchestrator=orchestrator,
    domain_repo=domain_repo,
)

# 列出可用工具
tools = server.list_tools()

# 调用工具
result = await server.call_tool(MCPToolCall(
    tool="generate_script",
    arguments={"talent_name": "李佳琦", "product_name": "口红"}
))

# 读取资源
resource = await server.read_resource("scriptagent://talent/T001")

# 渲染提示词
prompt = server.render_prompt("generate_opening", {...})
```

更多示例：`examples/mcp_server_example.py`

---

### A2A协议支持
**适用场景**：
- 多Agent协作（翻译Agent + 摘要Agent）
- 与其他框架互操作（OpenAI Swarm、AutoGen、LangChain）
- 构建Agent网络
- 标准化Agent通信

**兼容框架**：
- ✅ OpenAI Swarm（轻量级多Agent）
- ✅ Microsoft AutoGen（会话式Agent）
- ✅ LangChain Agent Protocol（工具调用）

**核心特性**：
- 标准化消息格式（支持text、JSON、tool_calls）
- 对话上下文管理（多轮对话、消息树）
- 协议互转（A2A ↔ OpenAI ↔ AutoGen ↔ LangChain）
- 错误处理和超时控制

**使用方法**：
```python
from script_agent.protocols import create_a2a_router_with_scriptagent

# 创建A2A路由器（自动注册所有ScriptAgent的Agent）
router = create_a2a_router_with_scriptagent(orchestrator)

# 列出所有Agent
agents = router.list_agents()

# 创建对话
conversation = router.create_conversation()

# 发送消息
message = A2AMessage(
    role=A2AAgentRole.USER,
    content={"talent_name": "李佳琦", "product_name": "口红"},
    sender="user_agent",
    receiver="orchestrator",
    conversation_id=conversation.id,
)

response = await router.send_message(message, timeout=10.0)

# 查看对话历史
history = conversation.to_openai_messages()
```

更多示例：`examples/a2a_protocol_example.py`

