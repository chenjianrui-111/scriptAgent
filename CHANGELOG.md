# Changelog

本文件用于持续维护 `script_agent` 的版本演进记录。  
记录格式参考 Keep a Changelog，版本号建议遵循 SemVer（`MAJOR.MINOR.PATCH`）。

## [Unreleased]

## [1.5.4] - 2026-02-19

### Added

- 新增完整监控栈与面板配置：
  - `docker-compose.monitoring.yml`（Prometheus + Grafana）
  - `monitoring/prometheus/prometheus.yml`（Prometheus 抓取配置）
  - `monitoring/grafana/provisioning/*`（数据源与仪表盘自动加载）
  - `monitoring/grafana/dashboards/script-agent-overview.json`（运行总览面板）
- 新增监控使用文档 `docs/MONITORING.md`，覆盖本地/线上启动、验收与安全建议。
- 新增监控回归测试 `tests/test_monitoring_metrics.py`，验证 `/metrics` 输出关键指标。
- 新增 HTTP 层监控埋点（`script_agent/api/app.py`）：
  - `script_agent_http_requests_total`
  - `script_agent_http_request_duration_seconds`
  - `script_agent_http_inflight_requests`
- 新增会话生命周期指标（`script_agent/observability/metrics.py` + `script_agent/services/session_manager.py`）：
  - `script_agent_session_events_total`
  - `script_agent_active_sessions`

### Changed

- 指标体系增强（`script_agent/observability/metrics.py`）：
  - 新增 `script_agent_app` 应用信息指标
  - 增加阶段耗时观测便捷方法（毫秒/秒）
- 编排链路补齐过程指标（`script_agent/agents/orchestrator.py`）：
  - 记录质量分与质量通过率
  - 回填 workflow 各阶段时延直方图
  - 记录生成计数（按 category/scenario）
- 配置样例补充 Grafana 账户变量（`.env.example`、`.env.production.example`）。
- README 补充监控面板部署入口与文档链接（`README.md`）。
- 监控镜像源调整（`docker-compose.monitoring.yml`）：
  - 采用 `m.daocloud.io/docker.io/*` 镜像，提升国内 ECS 拉取成功率。

### Fixed

- 修复租户隔离在多认证方式下的会话越权问题（`script_agent/api/auth.py`、`script_agent/api/app.py`）：
  - 认证优先级调整为 `Bearer > API Key > dev_bypass`
  - 引入统一 owner 作用域判定，JWT/dev/APIKey+X-User-Id 均按用户隔离
  - 修复开发环境下 Bearer 被 bypass 覆盖导致历史串读的问题
- 修复同租户同 API Key 场景下不同用户历史可见性问题（支持 `X-User-Id` 作用域隔离）。
- 补充租户隔离回归测试（`tests/test_auth_and_history_features.py`），覆盖开发/生产与混合认证场景。

## [1.5.3] - 2026-02-17

### Added

- 新增回归测试（`tests/test_agents.py`）：
  - `test_stream_skill_execution_failure_emits_error_token`
  - 覆盖 `skill_executing` 分支失败时流式输出必须返回 `[ERROR]`
- 新增回归测试（`tests/test_llm_client_reliability.py`）：
  - `test_clean_llm_response_strips_inline_prompt_echo_fragments`
  - 覆盖“语气保持一致/不要整段复述”等内联提示词回显清洗

### Changed

- 流式编排容错增强（`script_agent/agents/orchestrator.py`）：
  - `skill_executing` 路径在无有效脚本时不再静默结束
  - 统一显式返回 `[ERROR] <message>`，避免前端出现空白结果
- 前端流式交互增强（`script_agent/web/app.js`）：
  - 增加空流保护（仅空白 token 视为无效输出）
  - 空流时移除占位消息并展示错误提示
  - 错误分支清洗 `[ERROR]` 前缀，提升用户可读性
- 清洗与质检规则增强：
  - `script_agent/services/llm_client.py` 增加内联回显短语强清洗
  - `script_agent/agents/script_agent.py` 增加二次内联回显剥离与回显判定
  - `script_agent/agents/quality_agent.py` 扩展 prompt-echo 规则，识别“本轮要求/不要整段复述”等泄漏

### Fixed

- 修复多轮生成结果中夹带限制性提示词（如“语气保持一致，但信息表达要有新增，不要整段复述”）的问题。
- 修复换品等多轮场景下 `skill_executing` 失败时流式无输出导致前端“空气泡”的问题。

## [1.5.2] - 2026-02-17

### Added

- 新增“真实链路”API 级 E2E 测试（`tests/test_api_real_llm_e2e.py`）：
  - `/sessions -> /generate(旧商品) -> /generate(换品)`
  - 断言换品后文案必须包含新商品名，且不包含旧商品名
- 新增流式质量重试回归测试（`tests/test_agents.py`）：
  - 首轮输出句尾不完整时，触发质量反馈并自动重试
  - 第二轮返回完整文案
- 新增句尾完整性/回显清洗测试：
  - `test_structure_checker_detects_tail_incomplete`
  - `test_clean_llm_response_strips_prompt_echo_markdown_blocks`

### Changed

- 生成校验增强（`script_agent/agents/script_agent.py`）：
  - 增加“有效正文长度”判定（过滤提示词回显后再计数）
  - 增加“句尾完整性”判定（拦截“它采用了260g的”这类半句截断）
  - 回显命中/正文不足/句尾不完整场景统一进入强制重试
- 流式编排链路增强（`script_agent/agents/orchestrator.py`）：
  - 流式路径补齐质量校验闭环
  - 质量不通过时将反馈注入 `requirements` 并自动重试
  - 支持返回最佳降级结果并附带提示，避免直接中断
- 智谱模型输出预算动态提升（`script_agent/agents/script_agent.py`）：
  - 在 `zhipu` 后端下提高基础 `max_tokens`
  - 在回显/过短/句尾不完整重试场景进一步放大 token 预算
- 回显清洗规则扩展（`script_agent/services/llm_client.py`）：
  - 支持识别加粗字段、Markdown 标题、`本轮要求` 等行内回显形式

### Fixed

- 修复多轮生成中“正文未完成即结束”导致前端展示半句截断的问题。
- 修复流式路径此前仅按最小长度放行、未做质量校验导致不完整文案漏出的缺陷。

## [1.5.1] - 2026-02-18

### Added

- 新增 API 级换品端到端用例（`tests/test_auth_and_history_features.py`）：
  - `/sessions -> /generate(旧商品) -> /generate(换新品)`
  - 断言“换品后文案必须包含新商品名，且不再包含旧商品名”
- 新增换品意图/槽位回归测试（`tests/test_agents.py`）：
  - 续写换品时优先使用新商品槽位
  - 换品场景下意图从 `script_modification` 纠偏为 `script_generation`

### Changed

- 强化商品名抽取规则（`script_agent/agents/intent_agent.py`）：
  - 支持“换成/改成/介绍/推荐”等换品表达
  - 增加商品名噪声清洗（前后缀去噪，避免“卫龙辣条的卖点话术”误抽取）
- 意图识别流程调整（`script_agent/agents/intent_agent.py`）：
  - 显式抽取槽位优先于继承槽位，防止多轮对话中旧商品覆盖新商品
  - 增加 `_product_switch` / `_previous_product_name` / `_intent_adjusted` 标记
- 扩展 `script_generation` 工具入参 schema（`script_agent/skills/builtin/script_gen.py`）：
  - 接受换品内部标记字段，避免 preflight 被 `additionalProperties=false` 拦截

### Fixed

- 修复“多轮对话换品后仍沿用旧商品”导致文案错误的问题。
- 修复换品标记字段触发工具 schema 拦截，导致生成失败的问题。

## [1.5.0] - 2026-02-18

### Added

- 新增账号体系与认证接口：
  - `POST /api/v1/auth/register`
  - `POST /api/v1/auth/login`
  - 本地 SQLite 用户存储（`script_agent/api/auth.py`）
- 新增前端登录注册页：
  - `GET /auth`
  - `script_agent/web/auth.html`
  - `script_agent/web/auth.js`
- 新增会话历史删除能力：
  - `DELETE /api/v1/sessions/{session_id}`
  - 前端会话侧边栏删除按钮联动
- 新增达人/商品基础数据仓储：
  - `script_agent/services/domain_data_repository.py`
  - `ProfileAgent`、`ProductAgent` 支持优先读取数据库资料并回退原有构建逻辑
- 新增部署/配置样例与架构文档：
  - `.env.production.example`
  - `docs/architecture.mmd`
  - `docs/architecture_prompt_for_gemini.md`
- 新增测试覆盖：
  - `tests/test_auth_and_history_features.py`
  - `tests/test_domain_data_integration.py`
  - 前端 SSE 多行 chunk 兼容测试

### Changed

- 认证链路增强（`script_agent/api/auth.py`）：
  - 生产环境支持 Bearer JWT + API Key 鉴权
  - 鉴权上下文增加 `role` 并透传到会话快照
- API 层增强（`script_agent/api/app.py`）：
  - 新增 `GET /api/v1/frontend-config` 用于前端环境感知配置
  - 流式输出改为规范 SSE 多行 `data:` 编码，增强客户端兼容性
- 前端主页面（`script_agent/web/app.js`）：
  - 增加登录状态按钮（登录/退出）
  - 生产环境无 token 时引导跳转登录页
  - 支持从本地 token 自动注入 `Authorization` 请求头
- LLM 主后端可配置：
  - 新增 `LLM_PRIMARY_BACKEND`
  - 默认策略：`development/testing -> zhipu`，`production -> vllm`
- 配置样例扩展（`.env.example`）：
  - 新增 domain data / auth / frontend 暴露控制相关环境变量

### Fixed

- 修复流式接口在 chunk 含换行时的 SSE 格式兼容问题。
- 修复 `orchestrator` 在流式场景下“先输出后校验”导致短内容提示不稳定的问题（改为聚合后校验再输出）。

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
