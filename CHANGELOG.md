# Changelog

本文件用于持续维护 `script_agent` 的版本演进记录。  
记录格式参考 Keep a Changelog，版本号建议遵循 SemVer（`MAJOR.MINOR.PATCH`）。

## [Unreleased]

## [1.4.0] - 2026-02-17

### Added

- 新增多轮连贯性优化能力（`script_agent/agents/script_agent.py`）：
  - 会话目标摘要注入（任务/商品/最近诉求/上版核心）
  - 续写约束注入（禁止复用上版开头、要求新增信息点）
  - 重试修正提示注入（按失败原因动态补充修正指令）
- 新增生成后处理与校验能力：
  - 重复句去重
  - 字段清单/提示词泄漏抑制
  - 与历史话术高重合检测（含续写场景首句复用检测）
- 新增回归测试（`tests/test_agents.py`）：
  - `test_prompt_builder_injects_context_summary_and_continuation_constraints`
  - `test_script_generation_agent_retries_on_high_overlap`
  - `test_script_generation_agent_trims_reused_leading_sentence`

### Changed

- `PromptBuilder` 历史上下文窗口从最近2轮扩展为最近3轮，提升多轮信息覆盖度。
- `ScriptGenerationAgent._generate_with_retry` 增强为“生成后处理 -> 连贯性校验 -> 按原因重试 -> 兜底”链路。
- 在重试和兜底均失败但已有可用候选时，返回最佳软候选，避免直接空结果。

### Fixed

- 修复续写场景中容易复用上一版开头、导致“看起来不连贯/像重复生成”的问题。
- 修复部分输出包含字段化片段（如商品元信息清单）混入正文的问题。

## [1.3.0] - 2026-02-17

### Added

- 前端全面升级为科技高端双栏布局：
  - 新增会话侧边栏（历史记录列表、会话切换、折叠/展开）
  - 新增流程步进器（场景 → 品类 → 类型 → 商品 → 生成）
  - 新增连接状态指示器（idle/connected/generating/error 四态 + 脉冲动画）
  - 新增 Toast 通知系统（success/error/info 三种类型）
  - 新增话术操作按钮（复制/重新生成/导出 .txt）
  - 新增打字指示器（三点跳动动画）
  - 新增生成统计栏（耗时、token 数）
  - 新增 Canvas 动态点阵网格背景
  - 新增键盘快捷键（Ctrl+Enter 发送、Esc 关闭面板）
  - 新增响应式布局（< 768px 侧边栏自动折叠为 overlay）
- 新增 LLM 响应清洗模块 `clean_llm_response()`（`llm_client.py`）：
  - 移除 `<think>...</think>` 思考块
  - 按 `---话术正文---` 分隔符提取正文
  - 正则移除任意 `【...】` 方括号标题
  - 跳过前导 prompt 元数据行（`- 达人:`、`- 商品名:` 等）
  - 截断尾部 prompt 回显噪声
- 新增 Prompt 生成分隔符机制：在 prompt 末尾注入 `---话术正文---` 标记，明确指示模型正文起始位置
- 新增完整前端 E2E 测试套件（50 个测试），覆盖：
  - 连接配置面板、新建会话、发送按钮、快速回复芯片
  - 商品卡片（预设 + 自定义）、话术操作按钮、会话侧边栏
  - 流程步进器、Toast 系统、状态指示器、统计栏、键盘快捷键
  - 全点击流程（卖点/开场/种草/自定义商品/空输出/短输出/失败/重新生成/多轮对话）
  - LLM 响应清洗集成验证
- 新增 `clean_llm_response` 单元测试（10 个用例）：think 块、prompt 标题、商品标题、分隔符提取、尾部截断等

### Changed

- 前端三件套（`index.html`/`styles.css`/`app.js`）全面重构为双栏布局，新增 15+ CSS 组件和 16+ JS 模块
- `PromptBuilder._build_role_prompt()` 增加明确输出规则指令，禁止模型重复提示语或输出思考过程
- `ScriptGenerationAgent._generate_once()` 在同步和流式路径均应用 `clean_llm_response()` 后处理
- `Orchestrator.handle_stream()` 增加异常捕获和用户友好错误消息：
  - LLM 异常时返回 `[生成失败]` 提示
  - 内容过短时返回 `[提示] 生成内容过短` 提示
  - 内容为空时返回 `[生成失败] 未能生成有效话术` 提示

### Fixed

- 修复用户仅点击 UI 流程时生成的核心卖点话术内容为空的问题（流式接口先 yield 后验证的反模式）
- 修复 LLM 回显 prompt 中的 `【达人风格】`、`【商品信息】` 等中文方括号标题泄漏到前端的问题
- 修复小模型（Qwen 0.5b）输出 `<think>` 思考过程和 prompt 元数据混入话术正文的问题
- 修复第二轮对话因历史上下文注入被污染的首轮输出导致回显放大的问题

---

- 前端重构为对话引导式话术助手（深色主题 + 流式聊天气泡）：
  - 状态机驱动引导流程：场景选择（直播/短视频）→ 类型选择（开场白/卖点介绍）→ 商品选择 → 流式生成
  - 7 个预设商品卡片（美妆/食品/服饰）+ 自定义商品输入
  - 自动 session 管理，用户无需感知 session_id
  - 流式 token 实时写入聊天气泡，支持多轮追问
  - 深色背景 + 霓虹渐变光球 + 响应式布局
- `LLMServiceClient` 新增企业级可靠性控制层：
  - 可重试错误与不可重试错误分类（HTTP/网络/超时/载荷解析）
  - 分级超时（connect/read/total，区分 sync 与 stream）
  - 断路器（closed/open/half-open）
  - 请求幂等键透传（`Idempotency-Key`）和本地 in-flight 并发去重
  - 分层 fallback 规则（primary -> primary-general -> fallback-backend）
- 新增 `tests/test_llm_client_reliability.py`，覆盖重试、降级、幂等并发去重、断路器开启场景。
- 长期记忆检索新增混合召回能力（dense vector + sparse BM25）及轻量 rerank 流程，支持按场景/意图/商品动态调整 top-k。
- 新增会话记忆“任务相关性驱动”裁剪策略：基于 query/product/intent/recency 评分，优先保留高相关轮次。
- 新增工具调用安全模块 `script_agent/skills/security.py`：
  - strict JSON schema 校验器
  - tenant/role allowlist policy engine
  - prompt injection tripwire
- 新增工具调用安全测试用例（schema/allowlist/tripwire）。
- 新增配置测试 `tests/test_settings.py`，覆盖本地/生产环境 Qwen 默认模型分流与本地模型覆盖能力。
- 新增生成链路回归测试（`tests/test_agents.py`）：
  - 长期记忆提示词注入强度随 token 预算占比变化
  - script agent 报错/空文案时 skill 返回失败
- 新增远程部署工件：
  - `Dockerfile`
  - `docker-compose.prod.yml`
  - `.env.production.example`
  - `deploy/deploy_remote.sh`（SSH + rsync 一键部署）

### Changed

- 前端三件套（`index.html`/`styles.css`/`app.js`）全部重写，从控制台式 Debug UI 替换为用户友好的对话式交互界面。
- `tests/test_frontend_e2e.py` 断言更新以匹配新前端标识（`chat-area`/`advanceFlow`/`--bg-primary`）。
- `script_agent/services/llm_client.py` 全面增强：
  - vLLM/Ollama 请求统一补充状态检查与 `raise_for_status` 路径
  - 流式与同步调用都支持重试+降级策略
  - 健康检查改为显式校验 HTTP 成功状态
- `script_agent/config/settings.py` 增加 `LLM_*` 可靠性配置项（重试、超时、断路器、fallback 策略、幂等）。
- 流式接口 `POST /api/v1/generate/stream` 改为复用 `Orchestrator` 统一状态推进与 checkpoint writer，不再在 API 层手写固定状态。
- `Orchestrator.handle_stream` 对齐同步链路：支持 `checkpoint_loader/checkpoint_writer/checkpoint_saver`，checkpoint 中状态序列与审计字段统一（含 `PRODUCT_FETCHING`）。
- `script_agent/services/long_term_memory.py` 引入 hybrid fuse + rerank + adaptive top-k，memory/elasticsearch 后端统一支持 sparse 召回接口。
- `script_agent/services/session_manager.py` 裁剪策略升级为 relevance-aware：高相关旧轮次可被保留且避免过度压缩。
- `SkillRegistry` 升级为统一工具治理入口，所有 Skill 在执行前都经过 preflight（required slots + schema + policy + tripwire）。
- `Orchestrator` 的 skill 执行路径接入统一 preflight，拦截结果进入 checkpoint/audit 链路。
- `AuthContext` 增加 `role`，并在 API 层将角色注入会话快照供 tool policy 使用。
- 内置技能 (`script_gen`/`script_modify`/`batch_generate`) 新增严格输入 schema 定义。
- `script_agent/config/settings.py` 增加按环境的 Qwen 默认模型分流：
  - `development/testing` 默认 `qwen2.5:0.5b`
  - `production` 默认 `qwen2.5:7b`
  - 支持 `QWEN_MODEL_LOCAL` / `QWEN_MODEL_PRODUCTION` / `VLLM_MODEL` 覆盖
- `PromptBuilder` 的长期记忆提示词注入改为按 `longterm_token_budget/total_token_budget` 动态调节注入条数与片段长度。
- `Orchestrator` 在 skill 失败时透传错误信息到 API `error` 字段，便于前端与调用方定位问题。
- `README.md` 增加“远程服务器部署（Docker）”章节，包含一键部署与手动部署流程。

### Fixed

- 修复 script generation skill 在下游 agent 报错或返回空文案时可能误判为成功的问题。
- 修复本地开发环境默认模型名与已加载小模型不一致导致的 `model not found` 问题（通过环境化默认模型策略规避）。

## [1.2.0] - 2026-02-16

### Added

- 新增 `ProductAgent`（`script_agent/agents/product_agent.py`），支持商品信息理解、卖点补全和商品画像构建。
- 新增长期记忆检索模块 `script_agent/services/long_term_memory.py`：
  - embedding 向量化（`hash`/`sentence-transformers`）
  - 向量库检索（`memory`/`elasticsearch`）
  - 召回过滤（tenant/influencer/category/product）
- 编排新增 `PRODUCT_FETCHING` 状态，支持“画像 -> 商品 -> 生成”链路。
- 生成链路接入向量召回样本提示，支持“商品卖点 + 达人风格 + 历史高相关文案”联合生成。
- 新增会话 retention policy（保存时规则压缩 + 最大轮次裁剪 + 脚本记录裁剪）。

### Changed

- `ScriptGenerationAgent` 接入 `SessionContextCompressor`，多轮对话使用压缩会话记忆参与 Prompt 构建。
- `Orchestrator` 在成功生成后自动写回长期记忆，提升后续同商品/同达人场景命中率。
- `health` 接口新增长期记忆状态输出。
- 配置扩展：
  - `LONGTERM_MEMORY_*`（向量检索）
  - `SESSION_MAX_TURNS_PERSISTED` / `SESSION_COMPRESS_ON_SAVE` 等会话 retention 配置

### Fixed

- 修复商品场景下仅靠达人画像生成导致卖点信息缺失的问题。
- 修复会话历史持续增长导致上下文膨胀与存储压力累积的问题。

## [1.1.0] - 2026-02-16

### Added

- 新增会话分布式锁能力：`RedisSessionLockManager`（`SET NX PX` + Lua 原子释放），支持多实例一致并发控制。
- 新增 checkpoint 独立存储模块：`script_agent/services/checkpoint_store.py`。
- checkpoint 支持版本化写入、历史回放、审计字段（`trace_id/status/checksum/created_at`）。
- API 新增 checkpoint 查询接口：
  - `GET /api/v1/sessions/{session_id}/checkpoints`
  - `GET /api/v1/sessions/{session_id}/checkpoints/latest`
  - `GET /api/v1/sessions/{session_id}/checkpoints/{version}`
- 新增核心接口限流能力：`CoreRateLimiter`，支持 QPS + Token 双维度限制（Local/Redis）。
- 新增 LLM 主备降级能力：主模型失败后可切换备用模型（同步/流式首段失败均支持）。
- 健康检查新增并发锁、checkpoint、限流状态输出，便于运维巡检。

### Changed

- 编排器 `Orchestrator` 支持外置 checkpoint loader/writer，恢复与落盘逻辑从会话字段中解耦。
- 会话 `workflow_snapshot` 调整为“摘要信息”，完整执行状态转移到独立 checkpoint 存储。
- `generate` 和 `generate/stream` 接口统一接入核心限流保护和会话级并发锁。
- 配置体系扩展：
  - 编排：分布式锁开关、租期、重试间隔
  - checkpoint：存储类型、前缀、历史上限
  - 限流：后端、QPS、token/min
  - LLM：fallback 开关与后端参数

### Fixed

- 修复并发写同会话导致状态覆盖的风险（通过会话级锁串行化）。
- 修复中断恢复时对“截断脚本”复用的风险（恢复种子不复用不完整脚本内容）。
- 修复重复请求触发下游模型重复调用的问题（完成态去重缓存快速返回）。

## 维护约定

- 每次发版必须新增一个 `## [x.y.z] - YYYY-MM-DD` 小节。
- `Unreleased` 用于记录下一版本待发布变更。
- 条目建议按 `Added / Changed / Fixed` 分类维护。
