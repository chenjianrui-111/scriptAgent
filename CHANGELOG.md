# Changelog

本文件用于持续维护 `script_agent` 的版本演进记录。  
记录格式参考 Keep a Changelog，版本号建议遵循 SemVer（`MAJOR.MINOR.PATCH`）。

## [Unreleased]

### Added

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

### Changed

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

### Fixed

- 待补充

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
