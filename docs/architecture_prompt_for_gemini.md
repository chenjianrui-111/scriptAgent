# Gemini 架构图生成提示词

将以下提示词复制到 Gemini 中生成系统架构图。

---

## 提示词

请为我生成一张专业的企业级系统架构图，风格要求：深色科技风（Dark Tech Style），类似AWS/阿里云官方架构图的高端视觉效果。背景为深色（#0a0f1a），使用霓虹色渐变线条（紫→青→蓝），模块使用圆角卡片 + 微光阴影，整体布局从上到下分为6层。

## 系统名称
**ScriptAgent — 企业级多智能体话术生成系统**

## 分层架构（从上到下）

### 第1层：Frontend Layer（前端展示层）
紫色渐变卡片，图标用 🖥️
- **Chat UI**：双栏布局 + 深色主题 + 动态点阵背景
- **Session Sidebar**：历史会话列表 + 会话切换
- **Flow Stepper**：5步引导流程（场景 → 品类 → 类型 → 商品 → 生成）
- **SSE Stream Reader**：实时Token流式渲染
- **Script Actions**：复制 / 重新生成 / 导出
- 连接方式：HTTP REST + SSE（Server-Sent Events）

### 第2层：API Gateway（API网关层）
蓝色渐变卡片，图标用 🔒
- **FastAPI + Uvicorn ASGI**
- **Auth Middleware**：JWT / API-Key 认证，多租户隔离
- **Core Rate Limiter**：QPS + Token双维度限流（Local / Redis）
- **Session Lock**：分布式并发控制（Redis SET NX PX）
- 关键端点：POST /generate/stream（SSE）、POST /generate（同步）、POST /sessions
- **Prometheus /metrics** 端点

### 第3层：Orchestrator（编排调度层）
绿色渐变卡片，图标用 🔄
- **LangGraph State Machine**：状态驱动的工作流引擎
- 状态流转：INIT → INTENT_RECOGNIZING → PROFILE_FETCHING → PRODUCT_FETCHING → SCRIPT_GENERATING → QUALITY_CHECKING → COMPLETED
- 分支：低置信度 → CLARIFYING，质量不达标 → REGENERATING
- **Request Dedup Cache**：请求去重
- **Checkpoint Writer**：版本化存储 + 审计追踪 + 断点恢复
- **Skill Registry**：统一工具治理（Schema校验 + Policy + Prompt Injection Tripwire）

### 第4层：Multi-Agent System（多智能体层）
橙红色渐变卡片，每个Agent用独立小卡片，图标用 🤖
- **Intent Agent**：3级意图分类（TF-IDF → LogisticRegression → LLM Fallback）+ Slot抽取 + 指代消解
- **Profile Agent**：达人画像获取 + 风格提取（语气/口头禅/幽默度）+ LRU Cache
- **Product Agent**：商品信息理解 + 卖点补全 + 长期记忆召回
- **Script Agent**：Prompt工程（角色设定+风格约束+场景模板+商品信息+历史记忆）→ LLM流式生成 → clean_llm_response后处理
- **Quality Agent**：AC自动机敏感词检测 + 广告法合规校验 + 风格一致性评分
- Agent间通过 **AgentMessage** 协议通信（trace_id全链路追踪）

### 第5层：LLM Service Layer（大模型服务层）
金色渐变卡片，图标用 ⚡
- **LLMServiceClient**：统一接口，屏蔽后端差异
- **3级Fallback链路**：Primary → Primary-General → Fallback Backend
- **Circuit Breaker**：断路器（closed → open → half-open）
- **幂等去重**：Idempotency-Key + in-flight并发合并
- **指数退避重试**：3次重试，0.35s基础延迟，带随机抖动
- 后端：
  - **vLLM**（生产）：Qwen-7B，支持LoRA动态切换
  - **Ollama**（开发）：qwen2.5:0.5b / 7b
  - **Zhipu GLM**（兜底）：glm-4-flash

### 第6层：Memory & Storage（存储与记忆层）
灰蓝色渐变卡片，图标用 💾
- **Redis**：Session Store / Checkpoint Store / 分布式锁 / 限流计数器
- **SQLite**：Fallback持久化
- **Elasticsearch**：向量检索 + 长期记忆（Dense + BM25混合召回 + Rerank）
- **SentenceTransformers**：语义Embedding
- **Session Compressor**：多轮对话分级压缩（Zone A原文 / Zone B摘要 / Zone C丢弃）

## 横向贯穿模块（用虚线框在右侧）
- **Observability**：Prometheus Metrics + Distributed Tracing（trace_id全链路）
- **Config**：环境感知配置（dev/test/prod），支持环境变量覆盖
- **Security**：Strict JSON Schema校验 + Tenant/Role Allowlist + Prompt Injection防御

## 数据流箭头标注
- 前端 → API：`HTTP POST + SSE Stream`
- API → Orchestrator：`async/await`
- Orchestrator → Agents：`State-driven dispatch`
- Script Agent → LLM：`aiohttp stream`
- LLM → Script Agent：`Token stream + clean_llm_response()`
- Orchestrator → Redis：`Checkpoint + Session persist`
- Product Agent → Elasticsearch：`Vector similarity search`

## 视觉要求
1. 深色背景 #0a0f1a，卡片背景 #1a1a2e，边框带微光效果
2. 连接线使用渐变色（紫 #6C5CE7 → 青 #00cec9 → 蓝 #0984e3）
3. 每层用不同的主色调，形成彩虹渐变层次感
4. 关键路径（用户请求→生成→返回）用加粗发光线条突出
5. 右下角标注：ScriptAgent v1.3.0 | Multi-Agent Script Generation System
6. 整体比例 16:9，适合放在PPT或技术文档中
7. 字体清晰，中英文混排，关键术语用英文，描述用中文
