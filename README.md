# script_agent

企业级多 Agent 话术生成系统，已完成从“手工流程编排”向“LangGraph 图编排”的升级，并补齐了会话恢复、分布式并发控制、限流和模型降级能力。

当前版本见 `VERSION`，版本演进记录见 `CHANGELOG.md`。

## 架构总览

```text
Client / SDK
    |
FastAPI API Layer (`script_agent/api/app.py`)
    |
    +-- Auth & Tenant Isolation (`script_agent/api/auth.py`)
    +-- Core Rate Limit (QPS + Token) (`script_agent/services/core_rate_limiter.py`)
    +-- Session Lock (Local/Redis) (`script_agent/services/concurrency.py`)
    |
Orchestrator (`script_agent/agents/orchestrator.py`)
    |
    +-- LangGraph State Graph
    |     CONTEXT_LOADING -> INTENT_RECOGNIZING
    |       -> (INTENT_CLARIFYING | SKILL_EXECUTING | PROFILE_FETCHING)
    |       -> SCRIPT_GENERATING -> QUALITY_CHECKING -> COMPLETED/DEGRADED
    |
    +-- Skills Router (`script_agent/skills/`)
    +-- Sub Agents (`script_agent/agents/*.py`)
    |
Persistence & Infra
    +-- Session Store (Memory/Redis/SQLite) (`script_agent/services/session_manager.py`)
    +-- Workflow Checkpoint Store (Memory/Redis/SQLite) (`script_agent/services/checkpoint_store.py`)
    +-- Observability Metrics (`script_agent/observability/metrics.py`)
    +-- LLM Client + Fallback (`script_agent/services/llm_client.py`)
```

## 核心能力设计

### 1) LangGraph 编排与回退执行

- 编排器以 LangGraph 作为主执行引擎，状态由 `WorkflowState` 统一定义。
- 若环境未安装 LangGraph，自动降级到顺序执行路径，保持接口兼容。
- 对外接口保持不变：`handle_request`、`handle_stream`。

### 2) 对话快速恢复机制（Checkpoint 独立存储）

- Checkpoint 从 `SessionContext.workflow_snapshot` 拆分为独立存储层，支持 Memory/Redis/SQLite。
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
  - `RedisSessionLockManager`（分布式锁，`SET NX PX` + Lua 原子释放）
- 分布式锁异常时自动降级到本地锁，保证服务可用性。
- `/generate` 和 `/generate/stream` 均纳入会话锁，避免同会话并发写冲突。

### 4) 核心接口限流与保护

- `CoreRateLimiter` 支持 Local/Redis 两种后端。
- 对租户维度同时实施：
  - QPS 限流
  - Token/min 限流
- 超限返回 `429`，并附带 `reason` 与 `retry_after_seconds`。

### 5) 生成失败降级（LLM Fallback）

- 主模型调用失败时，`LLMServiceClient` 可自动切换到备用模型。
- 支持同步与流式场景（流式在首段失败时切换）。
- 可配置备用后端类型（vLLM/Ollama）和备用模型。

## 核心模块与功能映射

| 模块 | 文件 | 功能点 |
| --- | --- | --- |
| API 网关层 | `script_agent/api/app.py` | 会话创建、同步/流式生成、checkpoint 查询、健康检查、指标出口 |
| 鉴权层 | `script_agent/api/auth.py` | API Key/JWT 鉴权、租户隔离、基础限流 |
| 编排层 | `script_agent/agents/orchestrator.py` | LangGraph 图编排、状态推进、恢复续跑、去重缓存、checkpoint 写入 |
| 意图识别 | `script_agent/agents/intent_agent.py` | 意图分类、槽位提取、指代/续写消解、澄清判断 |
| 画像层 | `script_agent/agents/profile_agent.py` | 达人画像聚合、缓存利用、风格上下文补全 |
| 生成层 | `script_agent/agents/script_agent.py` | Prompt 构建、垂类话术生成、参数控制 |
| 质检层 | `script_agent/agents/quality_agent.py` | 敏感词、合规、风格一致性校验与重试建议 |
| Skill 扩展层 | `script_agent/skills/` | 意图到技能路由，支持生成/修改/批量等可插拔能力 |
| LLM 服务层 | `script_agent/services/llm_client.py` | 统一 vLLM/Ollama 接口、连接池复用、主备降级 |
| 会话管理层 | `script_agent/services/session_manager.py` | 会话增删改查、序列化、Memory/Redis/SQLite 多后端 |
| 并发控制层 | `script_agent/services/concurrency.py` | 本地锁/Redis 分布式会话锁、超时控制、统计 |
| Checkpoint 层 | `script_agent/services/checkpoint_store.py` | 版本化 checkpoint、回放、审计、存储后端抽象 |
| 限流层 | `script_agent/services/core_rate_limiter.py` | 租户维度 QPS + Token 双限流，支持 Redis 共享配额 |
| 可观测层 | `script_agent/observability/metrics.py` | 请求、延迟、锁超时、checkpoint 写入、缓存命中等指标 |
| 配置层 | `script_agent/config/settings.py` | 环境变量集中配置，覆盖编排/锁/限流/checkpoint/LLM 主备 |

## 对外 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/sessions` | 创建会话 |
| GET | `/api/v1/sessions` | 列出租户会话 |
| GET | `/api/v1/sessions/{session_id}` | 获取会话详情 |
| POST | `/api/v1/generate` | 同步生成 |
| POST | `/api/v1/generate/stream` | 流式生成（SSE） |
| GET | `/api/v1/sessions/{session_id}/checkpoints` | 查看 checkpoint 历史 |
| GET | `/api/v1/sessions/{session_id}/checkpoints/latest` | 查看最新 checkpoint |
| GET | `/api/v1/sessions/{session_id}/checkpoints/{version}` | 按版本回放 checkpoint |
| GET | `/api/v1/skills` | 查看已注册技能 |
| GET | `/api/v1/health` | 健康检查（含锁/限流/checkpoint 状态） |
| GET | `/metrics` | Prometheus 指标 |

### 示例

```bash
# 1) 创建会话
curl -X POST http://localhost:8080/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"influencer_name":"小雅","category":"美妆"}'

# 2) 同步生成（注意字段是 query）
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"session_id":"SESSION_ID","query":"帮我写一段618美妆直播开场话术，活泼一点"}'
```

## 关键配置（企业部署建议）

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `SESSION_STORE` | `memory` | 会话存储后端：`memory/redis/sqlite` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 地址（锁、缓存、限流等） |
| `DISTRIBUTED_LOCK_ENABLED` | `true` | 启用 Redis 分布式会话锁 |
| `CHECKPOINT_STORE` | `memory` | checkpoint 存储后端：`memory/redis/sqlite` |
| `CORE_RATE_LIMIT_ENABLED` | `true` | 启用核心接口限流 |
| `CORE_RATE_LIMIT_BACKEND` | `local` | 限流后端：`local/redis` |
| `CORE_RATE_QPS_PER_TENANT` | `8` | 租户 QPS 配额 |
| `CORE_RATE_TOKENS_PER_MIN` | `20000` | 租户分钟 token 配额 |
| `LLM_FALLBACK_ENABLED` | `true` | 主模型失败后启用备用模型 |
| `LLM_FALLBACK_BACKEND` | `ollama` | 备用后端类型：`ollama/vllm` |
| `LANGGRAPH_REQUIRED` | `false` | 是否强制要求 LangGraph 可用 |

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 开发模式
export APP_ENV=development
python -m script_agent.main

# 生产模式（示例）
export APP_ENV=production
export VLLM_BASE_URL=http://vllm-service:8000/v1
uvicorn script_agent.api.app:app --host 0.0.0.0 --port 8080 --workers 4
```

## 测试

当前测试文件：`tests/test_agents.py`（31 个测试函数，覆盖模型、状态机、编排器、恢复/并发/限流等核心能力）。

```bash
python -m pytest tests/ -v
```

## 后续迭代规范

- 版本变更统一记录到 `CHANGELOG.md`。
- README 仅维护“当前已实现架构与模块能力”，避免写入未落地设计。
- 新增核心模块时，必须同步更新：
  - `核心模块与功能映射`
  - `关键配置`
  - `对外 API`（若有新增接口）
