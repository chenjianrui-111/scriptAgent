# ScriptAgent

面向电商直播/短视频场景的多 Agent 话术生成系统。项目已经具备完整的前后端链路、鉴权、会话管理、流式生成、质量校验、长期记忆召回与多模型容灾能力，可直接本地运行，也可容器化部署到云服务器。

- 版本演进：`CHANGELOG.md`
- 阿里云部署手册：`DEPLOY_ALIYUN_ECS.md`
- 架构图与高保真生成提示词：`docs/architecture.mmd`、`docs/architecture_prompt_for_gemini.md`

---

## 一、项目定位

ScriptAgent 解决的是“可持续、多轮可控、可上线运营”的话术生成，不是单次问答 Demo。核心目标：

- 多轮对话中保持意图连续，支持换品/换风格/续写。
- 输出可直接口播，避免提示词回显、半句截断、空结果。
- 线上具备基本工程能力：鉴权、限流、并发锁、checkpoint、降级容灾。

---

## 二、系统架构（简版）

```text
Web UI / API Client
        |
FastAPI API Layer
(app/auth/sessions/generate/stream/health/metrics)
        |
Orchestrator (LangGraph / fallback sequential)
        |
+-- Intent Agent      (意图识别 + 槽位抽取 + 换品纠偏)
+-- Profile Agent     (缓存 + 数据库 + 通用画像回退)
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
```

---

## 三、核心能力设计（按模块）

### 1) API 层（`script_agent/api/app.py`）
**出色点**
- 同时支持同步生成与 SSE 流式生成，便于前端实时渲染。
- 全链路注入 `trace_id`、状态与时延，便于排障。
- 会话级并发锁保护，避免同会话并发生成造成数据污染。
- 健康接口返回编排、锁、限流、checkpoint、长期记忆等运行状态。

### 2) 鉴权与租户隔离（`script_agent/api/auth.py`）
**出色点**
- 支持 `X-API-Key` 与 `Bearer JWT` 双认证模式。
- 所有核心会话接口按 `tenant_id` 做强隔离。
- 内置登录注册能力（SQLite 用户库），可直接用于内网运营后台。
- 开发环境支持 bypass，降低本地联调门槛。

### 3) 编排中枢（`script_agent/agents/orchestrator.py`）
**出色点**
- LangGraph 声明式状态图 + 顺序回退执行双通道，鲁棒性更好。
- 支持请求去重缓存（完成态快速返回），降低重复请求成本。
- 流式与同步链路统一状态推进、checkpoint 写入与错误语义。
- Skill 路由与 Agent 链路共存，可逐步扩展工具化能力。

### 4) 意图识别 Agent（`script_agent/agents/intent_agent.py`）
**出色点**
- 分级意图识别：快速分类优先，低置信再走 LLM 兜底。
- 多来源槽位合并：规则抽取 + 指代消解 + 上下文继承。
- 自动写入 `_raw_query/requirements`，让生成器拿到原始用户诉求。
- 换品纠偏逻辑完善：自动识别“从旧商品切到新商品”，并将意图纠正到生成链路。

### 5) 达人画像 Agent（`script_agent/agents/profile_agent.py`）
**出色点**
- L1 本地缓存 + L2 缓存 + DB + 通用画像回退，多级容错。
- 支持按达人/品类动态构建画像，不依赖单一数据源。
- 在数据缺失时仍能给出可用风格配置，避免链路中断。

### 6) 商品 Agent（`script_agent/agents/product_agent.py`）
**出色点**
- DB 命中优先，规则知识库兜底，确保商品画像始终可构建。
- 自动融合：商品特征、卖点、合规提醒、目标人群。
- 内置长期记忆召回查询拼装，提升多轮场景下的话术一致性与信息密度。

### 7) 话术生成 Agent（`script_agent/agents/script_agent.py`）
**出色点**
- PromptBuilder 按“角色/风格/场景/商品/历史/记忆/要求”结构化组装。
- 强制最小有效长度、句尾完整性、重复抑制、换品约束校验。
- 主模型多次重试 + 失败后自动走 fallback 模型。
- 输出后置清洗增强：字段回显、提示词回显、占位符、重复句等自动剥离。

### 8) 质量校验 Agent（`script_agent/agents/quality_agent.py`）
**出色点**
- 并行检测敏感词、合规、风格一致性、结构完整性。
- 内置 AC 自动机敏感词扫描，效率和可解释性兼顾。
- 对 prompt-echo、内容过简、句尾不完整、换品违规进行结构化反馈。
- 反馈可回注到重试要求，形成“生成-质检-再生成”闭环。

### 9) LLM 客户端（`script_agent/services/llm_client.py`）
**出色点**
- 统一封装 `zhipu/vllm/ollama`，减少上层分支复杂度。
- 内置重试、断路器、超时分层、fallback 计划。
- 支持流式/同步一致清洗逻辑，显著降低脏输出。
- 连接池复用，减少高并发场景连接开销。

### 10) 会话与状态基础设施

#### 会话存储（`script_agent/services/session_manager.py`）
**出色点**
- `memory/redis/sqlite` 三后端统一抽象。
- 会话裁剪与压缩策略，控制多轮对话体积。

#### Checkpoint（`script_agent/services/checkpoint_store.py`）
**出色点**
- 版本化保存，支持 latest / history / replay。
- 每条记录带 checksum 与状态，便于审计与恢复。

#### 并发锁（`script_agent/services/concurrency.py`）
**出色点**
- 支持 local/redis 分布式会话锁。
- 锁服务异常可降级本地锁，服务不中断。

#### 核心限流（`script_agent/services/core_rate_limiter.py`）
**出色点**
- 租户维度双限流：QPS + token/min。
- local 与 redis 两种配额后端，适配单机与多实例。

### 11) Skill 治理与安全（`script_agent/skills/`）
**出色点**
- SkillRegistry 按意图评分路由，可插拔扩展生成/修改/批量技能。
- 统一 preflight 安全链路：required slots + strict schema + role/tenant allowlist + prompt injection tripwire。
- 工具调用治理能力独立于业务逻辑，适合持续扩展。

### 12) 领域数据仓储（`script_agent/services/domain_data_repository.py`）
**出色点**
- 达人画像与商品画像同库管理，默认 SQLite，部署简单。
- 自动建表与 upsert，便于从 mock 平滑迁移到真实数据。
- Profile/Product Agent 无缝集成，命中后可直接提升生成质量。

### 13) 前端交互（`script_agent/web/`）
**出色点**
- 双栏对话式 UI：会话历史 + 主对话区。
- 支持登录/注册页面与主页面联动。
- 生成链路含空流保护、错误提示、重生成、导出、复制。
- 线上可关闭“连接配置面板”，避免直接暴露调试入口。

---

## 四、功能清单

- 话术生成：开场话术、卖点介绍、促销话术、种草文案。
- 多轮对话：续写、改风格、换商品、重生成。
- 历史会话：创建、查询、删除。
- 认证能力：注册、登录、API Key/JWT。
- 质量治理：敏感词、合规、结构完整性、换品约束。
- 模型治理：主模型重试、兜底模型切换、回显清洗。

---

## 五、快速开始（本地）

### 1) 环境要求

- Python `>=3.10`
- 推荐：`pip`、`virtualenv`

### 2) 安装与配置

```bash
git clone <your-repo-url>
cd scriptAgent

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

pip install -r requirements.txt
cp .env.example .env
```

### 3) 启动服务

```bash
python -m script_agent.main
# 或
uvicorn script_agent.api.app:app --host 127.0.0.1 --port 8080
```

访问：
- 主页面：`http://127.0.0.1:8080/`
- 登录页：`http://127.0.0.1:8080/auth`
- 健康检查：`http://127.0.0.1:8080/api/v1/health`

---

## 六、关键环境变量

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

## 七、API 概览

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

## 八、部署

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

---

## 九、测试

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

## 十、目录结构（核心）

```text
script_agent/
  api/            # FastAPI 接口、鉴权
  agents/         # Intent/Profile/Product/Script/Quality/Orchestrator
  skills/         # Skill 框架、内置技能、安全治理
  services/       # LLM 客户端、会话/锁/限流/checkpoint/长期记忆/领域数据
  models/         # 数据结构与状态定义
  web/            # 前端页面（主页面 + 登录页）
  config/         # 配置中心

tests/            # 单元/集成/E2E 测试
```

---

## 十一、补充说明

- 项目默认可从 mock/规则能力启动，随后平滑接入真实达人与商品数据。
- 如果你要做线上版本，建议优先完成三件事：
  - 关闭前端连接配置面板（`FRONTEND_EXPOSE_CONNECTION_PANEL=false`）
  - 启用真实鉴权策略（API Key/JWT）
  - 使用 Redis 作为锁与限流后端，避免多实例配额不一致
