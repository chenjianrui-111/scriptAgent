# 话术助手 Agent 系统

基于 LoRA 微调的多 Agent 协作系统，为电商达人自动生成垂类定制话术，支持多轮对话和实时迭代优化。

## 项目结构

```
script_agent/
├── models/                      # 数据模型层
│   ├── message.py               # AgentMessage 标准化消息协议 (同步/异步/流式)
│   ├── state_machine.py         # WorkflowState 状态机 (INIT→INTENT→PROFILE→SCRIPT→QUALITY→COMPLETED)
│   └── context.py               # 三层上下文: SystemContext / SessionContext / InfluencerProfile
│
├── agents/                      # Agent 层
│   ├── base.py                  # BaseAgent 抽象基类 (统一调用入口, 计时, 异常处理)
│   ├── intent_agent.py          # 意图识别 Agent — 三级分级 + 槽位提取 + 指代消解
│   ├── profile_agent.py         # 达人画像 Agent — 多级缓存 (L1 内存 / L2 Redis / L3 实时构建)
│   ├── script_agent.py          # 话术生成 Agent — LoRA 推理 + Prompt 工程 + 敏感词过滤
│   ├── quality_agent.py         # 质量校验 Agent — 敏感词/合规/风格一致性 多维并行检测
│   └── orchestrator.py          # 编排器 — 状态机驱动四个子 Agent 协作
│
├── context/                     # 上下文管理层
│   ├── system_context.py        # 系统层上下文 (模板化 Prompt, Prefix Caching 友好)
│   └── session_compressor.py    # 会话层分级压缩 (Zone A/B/C 策略, Token 预算动态分配)
│
├── services/                    # 服务层
│   ├── llm_client.py            # 统一 LLM 调用接口 (屏蔽 vLLM / Ollama 差异)
│   ├── session_manager.py       # 会话生命周期管理 (内存 / Redis)
│   └── style_extractor.py       # 风格提取 (口头禅识别 + 正式度分析 + 画像融合)
│
├── config/
│   └── settings.py              # 配置管理 (环境变量驱动, dev/test/prod 切换)
│
├── api/
│   └── app.py                   # FastAPI REST API (同步 + SSE 流式)
│
├── tests/
│   └── test_agents.py           # 23 个测试用例, 覆盖所有核心组件
│
├── main.py                      # 启动入口
└── requirements.txt             # 依赖项
```

## 核心架构

### 多 Agent 协作流程

```
用户输入
  │
  ▼
┌─────────────────────────────────────────┐
│           Orchestrator (编排器)           │
│  状态机驱动: INIT→INTENT→PROFILE→SCRIPT→QUALITY→DONE  │
└───┬──────────┬──────────┬──────────┬────┘
    │          │          │          │
    ▼          ▼          ▼          ▼
 Intent     Profile    Script    Quality
  Agent      Agent     Agent      Agent
  │          │          │          │
  │ 三级分级  │ 多级缓存  │ LoRA推理  │ 多维检测
  │ 槽位提取  │ L1/L2/L3 │ Prompt工程│ 敏感词+合规
  │ 指代消解  │ 画像构建  │ 流式生成  │ 风格一致性
```

### 三层上下文管理

| 层级 | 内容 | Token 预算 | 更新频率 |
|------|------|-----------|---------|
| System | 角色设定 + 垂类知识 + 合规约束 | ~500 | 不变 (Prefix Caching) |
| Session | 对话历史 + 实体缓存 + 槽位状态 | ~1800 | 每轮更新 |
| Long-term | 达人画像 + 风格偏好 | ~800 | 跨会话持久化 |

### 会话分级压缩

| 区域 | 范围 | 策略 | 信息保留率 |
|------|------|------|-----------|
| Zone A | 最近 2 轮 | 完整保留 | 100% |
| Zone B-1 | 3-4 轮前 | LLMLingua-2 精细压缩 | 88% |
| Zone B-2 | 5-6 轮前 | 规则压缩 | 70% |
| Zone C | 7 轮以前 | LLM 深度摘要 | ~30% |

## 快速启动

### 开发环境 (Ollama)

```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export APP_ENV=development
export OLLAMA_BASE_URL=http://localhost:11434

# 启动
python -m script_agent.main
```

### 生产环境 (vLLM)

```bash
export APP_ENV=production
export VLLM_BASE_URL=http://vllm-service:8000/v1

uvicorn script_agent.api.app:app --host 0.0.0.0 --port 8080 --workers 4
```

### 运行测试

```bash
# 使用 pytest
python -m pytest tests/ -v

# 不依赖 pytest
python tests/test_agents.py
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/sessions` | 创建会话 |
| POST | `/api/v1/generate` | 生成话术 (同步) |
| POST | `/api/v1/generate/stream` | 生成话术 (流式 SSE) |
| GET  | `/api/v1/sessions` | 列出会话 |
| GET  | `/api/v1/sessions/{id}` | 获取会话详情 |
| GET  | `/api/v1/health` | 健康检查 |

### 示例请求

```bash
# 创建会话
curl -X POST http://localhost:8080/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "t1", "influencer_name": "小雅"}'

# 生成话术
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "message": "帮我写一段美妆直播开场话术，618大促的，活泼一点"}'
```

## 技术亮点

1. **LoRA 微调方案** — vLLM 动态切换 adapter / Ollama 合并模型, 统一接口屏蔽差异
2. **Multi-Agent 编排** — 编排器模式 + 状态机驱动 + 消息协议解耦
3. **三层上下文** — 系统层 Prefix Caching + 会话层分级压缩 + 长期记忆向量检索
4. **分级压缩** — Zone A/B/C 策略, Token 预算动态分配, 信息保留 88%
5. **多轮对话** — 实体追踪、指代消解 (实体/内容/隐式继承)
6. **质量校验** — 敏感词 + 合规 + 风格一致性并行检测, 支持重试+降级
7. **风格提取** — 规则 + NLP 统计分析, 加权融合 (新 30% + 旧 70%) 更新画像

## 依赖项

- **核心**: fastapi, uvicorn, pydantic, aiohttp
- **NLP**: jieba (中文分词)
- **可选**: transformers, torch, peft (LoRA), onnxruntime, redis, pymysql
